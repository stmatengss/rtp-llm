"""End-to-end Qwen2.5-Omni benchmark: thinker → talker → token2wav.

Usage:
    CUDA_VISIBLE_DEVICES=5 python bench_omni_e2e.py \
        --ckpt /root/models/Qwen/Qwen2.5-Omni-7B \
        --thinker-url http://localhost:18080 \
        --prompt "Tell me a joke." \
        --speaker Chelsie \
        --output output.wav
"""
import argparse
import json
import logging
import os
import struct
import time
from typing import List, Optional, Tuple

import requests
import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("bench_omni_e2e")


def load_thinker_embeddings(ckpt_path: str, device: str = "cpu") -> torch.nn.Embedding:
    """Load thinker embedding layer from safetensors."""
    from safetensors.torch import load_file

    index_path = os.path.join(ckpt_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    embed_key = "thinker.model.embed_tokens.weight"
    shard_file = index["weight_map"].get(embed_key)
    if shard_file is None:
        raise RuntimeError(f"Cannot find {embed_key} in weight index")

    logger.info(f"Loading thinker embeddings from {shard_file}...")
    shard_path = os.path.join(ckpt_path, shard_file)
    weights = load_file(shard_path)
    embed_weight = weights[embed_key]
    logger.info(f"Thinker embedding shape: {embed_weight.shape}")

    vocab_size, embed_dim = embed_weight.shape
    embedding = torch.nn.Embedding(vocab_size, embed_dim)
    embedding.weight = torch.nn.Parameter(embed_weight)
    embedding = embedding.to(device=device, dtype=torch.bfloat16)
    embedding.eval()
    return embedding


def load_speaker_data(ckpt_path: str, speaker: str, device: str = "cpu"):
    """Load speaker conditioning data from spk_dict.pt."""
    spk_path = os.path.join(ckpt_path, "spk_dict.pt")
    spk_dict = torch.load(spk_path, map_location=device)

    if speaker not in spk_dict:
        available = list(spk_dict.keys())
        raise ValueError(f"Speaker '{speaker}' not found. Available: {available}")

    spk = spk_dict[speaker]
    return {
        "bos_token": spk["bos_token"],
        "cond": spk["cond"].float().to(device),
        "ref_mel": spk["ref_mel"].float().to(device),
    }


def tokenize_prompt(prompt: str, thinker_url: str) -> List[int]:
    """Tokenize prompt using the server's tokenizer endpoint."""
    resp = requests.post(
        f"{thinker_url}/tokenize",
        json={"prompt": prompt},
    )
    if resp.status_code == 200:
        data = resp.json()
        return data.get("token_ids", data.get("tokens", []))

    # Fallback: use transformers tokenizer
    logger.warning("Tokenize endpoint not available, using transformers tokenizer")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(os.environ.get("CHECKPOINT_PATH", ""), ".."),
        trust_remote_code=True,
    )
    return tokenizer.encode(prompt)


def call_thinker_streaming(
    prompt: str,
    thinker_url: str,
    max_new_tokens: int = 256,
) -> Tuple[str, List[int], List[List[float]]]:
    """Call thinker engine with streaming to collect per-token hidden states.

    Returns:
        text: generated text
        token_ids: list of generated token IDs
        hidden_states: list of [3584] hidden state vectors (one per generated token)
    """
    logger.info(f"Calling thinker (streaming, max_tokens={max_new_tokens})...")

    resp = requests.post(
        thinker_url,
        json={
            "prompt": prompt,
            "generate_config": {
                "return_hidden_states": True,
                "max_new_tokens": max_new_tokens,
            },
            "stream": True,
        },
        stream=True,
    )
    resp.raise_for_status()

    generated_text_parts = []
    all_hidden_states = []
    all_token_ids = []
    prev_text = ""

    for line in resp.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8")
        if line_str.startswith("data: "):
            line_str = line_str[6:]

        try:
            data = json.loads(line_str)
        except json.JSONDecodeError:
            continue

        # Collect hidden states
        hs = data.get("hidden_states")
        if hs is not None and len(hs) > 0:
            all_hidden_states.append(hs[0])

        # Collect text incrementally
        text = data.get("response", "")
        if text and text != prev_text:
            generated_text_parts.append(text[len(prev_text):] if len(text) > len(prev_text) else text)
            prev_text = text

    full_text = prev_text
    logger.info(f"Thinker generated: '{full_text[:80]}...' ({len(all_hidden_states)} tokens)")
    return full_text, all_token_ids, all_hidden_states


def call_thinker_non_streaming(
    prompt: str,
    thinker_url: str,
    max_new_tokens: int = 256,
) -> Tuple[str, List[List[float]]]:
    """Call thinker without streaming — gets hidden states for last token only.

    For e2e, we need per-token hidden states, so this is a fallback.
    """
    resp = requests.post(
        thinker_url,
        json={
            "prompt": prompt,
            "generate_config": {
                "return_hidden_states": True,
                "return_all_hidden_states": True,
                "max_new_tokens": max_new_tokens,
            },
        },
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", ""), data.get("hidden_states", [])


def save_wav(waveform: torch.Tensor, path: str, sample_rate: int = 24000):
    """Save waveform tensor to WAV file."""
    audio = waveform.squeeze().cpu().float().numpy()
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    with open(path, "wb") as f:
        num_samples = len(audio_int16)
        data_size = num_samples * 2  # 16-bit
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))  # PCM
        f.write(struct.pack("<H", 1))  # mono
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * 2))
        f.write(struct.pack("<H", 2))  # block align
        f.write(struct.pack("<H", 16))  # bits per sample
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(audio_int16.tobytes())

    logger.info(f"Saved WAV: {path} ({num_samples/sample_rate:.2f}s, {sample_rate}Hz)")


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni end-to-end benchmark")
    parser.add_argument("--ckpt", required=True, help="Model checkpoint path")
    parser.add_argument("--thinker-url", default="http://localhost:18080", help="Thinker server URL")
    parser.add_argument("--prompt", default="Tell me a short joke.", help="Input prompt")
    parser.add_argument("--speaker", default="Chelsie", help="Speaker name")
    parser.add_argument("--output", default="output.wav", help="Output WAV file path")
    parser.add_argument("--max-thinker-tokens", type=int, default=256, help="Max thinker tokens")
    parser.add_argument("--max-talker-tokens", type=int, default=4096, help="Max talker tokens")
    parser.add_argument("--device", default="cuda:0", help="Device for talker/token2wav")
    args = parser.parse_args()

    device = args.device
    t0 = time.time()

    # ============================================================
    # Step 1: Load talker, token2wav, and thinker embeddings
    # ============================================================
    logger.info("=== Loading models ===")

    from rtp_llm.omni.models.qwen2_5_omni.talker_inference import TalkerInference
    from rtp_llm.omni.models.qwen2_5_omni.token2wav_model import Token2WavModel

    t_load_start = time.time()
    talker = TalkerInference.from_pretrained(args.ckpt, device=device)
    t_talker_loaded = time.time()
    logger.info(f"Talker loaded in {t_talker_loaded - t_load_start:.1f}s")

    token2wav = Token2WavModel.from_pretrained(args.ckpt, device=device)
    t_t2w_loaded = time.time()
    logger.info(f"Token2Wav loaded in {t_t2w_loaded - t_talker_loaded:.1f}s")

    thinker_embed = load_thinker_embeddings(args.ckpt, device=device)
    t_embed_loaded = time.time()
    logger.info(f"Thinker embeddings loaded in {t_embed_loaded - t_t2w_loaded:.1f}s")

    speaker_data = load_speaker_data(args.ckpt, args.speaker, device=device)
    logger.info(f"Speaker: {args.speaker}, BOS token: {speaker_data['bos_token']}")

    # ============================================================
    # Step 2: Run thinker (via rtp-llm server)
    # ============================================================
    logger.info("=== Running thinker ===")
    t_thinker_start = time.time()

    text, _, per_token_hidden_states = call_thinker_streaming(
        args.prompt, args.thinker_url, args.max_thinker_tokens,
    )

    t_thinker_end = time.time()
    logger.info(f"Thinker time: {t_thinker_end - t_thinker_start:.2f}s, "
                f"generated {len(per_token_hidden_states)} tokens")
    logger.info(f"Generated text: {text}")

    if not per_token_hidden_states:
        logger.error("No hidden states returned from thinker! Cannot proceed.")
        return

    # ============================================================
    # Step 3: Prepare talker inputs from thinker outputs
    # ============================================================
    logger.info("=== Preparing talker inputs ===")

    # Convert hidden states to tensors
    # Each entry in per_token_hidden_states is a [3584] float list
    hidden_dim = len(per_token_hidden_states[0])
    num_tokens = len(per_token_hidden_states)

    # For the prompt, we don't have per-token hidden states from streaming
    # (streaming only returns hidden states for generated tokens).
    # The HF approach uses the prompt's hidden states too, but since
    # we're using the C++ engine, we'll use a simplified approach:
    # - Use a zero vector for the prompt hidden state
    # - Use generated token hidden states for the reply part

    # Tokenize the prompt to get input_ids
    # Use a simple approach: encode via the server
    prompt_text = args.prompt

    # Create mock prompt hidden states (zeros) — the prompt hidden states
    # are used for position encoding but the main signal comes from the
    # generated token hidden states
    prompt_len = 1  # simplified: treat prompt as single token for talker

    # Build thinker_hidden_states and thinker_token_embeds
    # Entry [0] = prompt, entries [1:] = generated tokens
    dtype = torch.bfloat16

    # For generated tokens, we need both last-layer hidden states and first-layer embeddings
    # hidden_states come from the C++ engine (last layer)
    # token_embeds come from running embed_tokens on the generated token IDs

    # We need to recover the generated token IDs from the text
    # Use the tokenizer to encode the generated text
    try:
        resp = requests.post(
            f"{args.thinker_url}/tokenize",
            json={"prompt": text},
        )
        if resp.status_code == 200:
            gen_token_ids = resp.json().get("token_ids", [])
        else:
            raise RuntimeError("Tokenize endpoint failed")
    except Exception:
        logger.warning("Could not tokenize generated text, using fallback")
        gen_token_ids = list(range(num_tokens))

    logger.info(f"Generated token IDs count: {len(gen_token_ids)}, hidden states count: {num_tokens}")

    # Align token count: hidden states may differ slightly from tokenized text
    # Use min of both
    effective_tokens = min(len(gen_token_ids), num_tokens)
    gen_token_ids = gen_token_ids[:effective_tokens]

    # Build prompt hidden state (zeros for now)
    prompt_input_ids = torch.tensor([[0]], dtype=torch.long, device=device)
    prompt_hs = torch.zeros(1, 1, hidden_dim, dtype=dtype, device=device)
    prompt_embed = torch.zeros(1, 1, hidden_dim, dtype=dtype, device=device)

    thinker_hidden_states = [prompt_hs]
    thinker_token_embeds = [prompt_embed]

    # Add per-generated-token hidden states and embeddings
    gen_ids_tensor = torch.tensor(gen_token_ids, dtype=torch.long, device=device)
    gen_embeds = thinker_embed(gen_ids_tensor)  # [num_tokens, 3584]

    for i in range(effective_tokens):
        hs_vec = torch.tensor(per_token_hidden_states[i], dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        embed_vec = gen_embeds[i:i+1].unsqueeze(0).to(dtype)
        thinker_hidden_states.append(hs_vec)
        thinker_token_embeds.append(embed_vec)

    # ============================================================
    # Step 4: Run talker
    # ============================================================
    logger.info("=== Running talker ===")
    t_talker_start = time.time()

    codec_tokens = talker.generate(
        thinker_hidden_states=thinker_hidden_states,
        thinker_token_embeds=thinker_token_embeds,
        input_ids=prompt_input_ids,
        speaker_bos_token=speaker_data["bos_token"],
        thinker_embed_tokens=thinker_embed,
        max_new_tokens=args.max_talker_tokens,
        temperature=0.9,
        top_k=40,
        top_p=0.8,
        repetition_penalty=1.05,
    )

    t_talker_end = time.time()
    num_codec = codec_tokens.shape[1]
    logger.info(f"Talker time: {t_talker_end - t_talker_start:.2f}s, "
                f"generated {num_codec} codec tokens")
    logger.info(f"Codec tokens (first 20): {codec_tokens[0, :20].tolist()}")

    if num_codec == 0:
        logger.error("No codec tokens generated! Cannot produce audio.")
        return

    # Filter out special tokens (EOS, PAD)
    codec_tokens_filtered = codec_tokens[codec_tokens < 8192].unsqueeze(0)
    if codec_tokens_filtered.shape[1] == 0:
        logger.error("All codec tokens were special tokens. No audio to generate.")
        return
    logger.info(f"Codec tokens after filtering: {codec_tokens_filtered.shape[1]}")

    # ============================================================
    # Step 5: Run token2wav
    # ============================================================
    logger.info("=== Running token2wav ===")
    t_t2w_start = time.time()

    with torch.no_grad():
        waveform = token2wav(
            codec_tokens_filtered.to(device),
            conditioning=speaker_data["cond"],
            reference_mel=speaker_data["ref_mel"],
        )

    t_t2w_end = time.time()
    logger.info(f"Token2Wav time: {t_t2w_end - t_t2w_start:.2f}s")
    logger.info(f"Waveform shape: {waveform.shape}")

    # ============================================================
    # Step 6: Save output
    # ============================================================
    save_wav(waveform, args.output, sample_rate=24000)

    total_time = time.time() - t0
    logger.info("=== Summary ===")
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Generated text: {text}")
    logger.info(f"Thinker tokens: {effective_tokens}")
    logger.info(f"Codec tokens: {num_codec}")
    logger.info(f"Audio duration: {waveform.numel() / 24000:.2f}s")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"  Model loading: {t_embed_loaded - t_load_start:.2f}s")
    logger.info(f"  Thinker: {t_thinker_end - t_thinker_start:.2f}s")
    logger.info(f"  Talker: {t_talker_end - t_talker_start:.2f}s")
    logger.info(f"  Token2Wav: {t_t2w_end - t_t2w_start:.2f}s")


if __name__ == "__main__":
    main()
