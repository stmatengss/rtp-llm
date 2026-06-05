"""Validate that REAL thinker hidden states produce different audio than RANDOM.

The hypothesis: if the talker is truly conditioned on thinker hidden states,
then feeding it real states (corresponding to "Why was the math book sad...")
vs random states should produce measurably different codec tokens and audio.
If feeding random states produces the SAME audio, the talker is ignoring
the conditioning (i.e., our pipeline isn't actually using it).

This test:
  1. Loads thinker + talker via RTP engines
  2. Generates codec tokens with REAL thinker hidden states (the e2e path)
  3. Generates codec tokens with RANDOM hidden states (matching shape)
  4. Asserts the two codec sequences differ substantially
  5. Saves both WAVs side-by-side for manual A/B listening

No torchaudio / HuggingFace dependencies.

Usage:
    CUDA_VISIBLE_DEVICES=5,6 python test_audio_validation.py
"""
import gc
import inspect
import logging
import os
import struct
import sys
import time

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger("test_audio_validation")

CKPT = "/root/models/Qwen/Qwen2.5-Omni-7B"
PROMPT = "Tell me a short joke."
TALKER_CODEC_BOS = 8293
TALKER_CODEC_EOS = 8294
THINKER_HIDDEN_SIZE = 3584

OUT_REAL = "/root/test_audio_real.wav"
OUT_RAND = "/root/test_audio_random.wav"

# Thresholds: if codec token sequences agree on >75% of positions, the
# conditioning isn't doing much. We expect substantial divergence.
MAX_AGREEMENT_RATIO = 0.75


def make_engine_config(kv_cache_mb=2048):
    from rtp_llm.ops import (
        ParallelismConfig, RuntimeConfig, FMHAConfig, DeviceResourceConfig,
        MoeConfig, NcclCommConfig, PDSepConfig, ConcurrencyConfig,
        ProfilingDebugLoggingConfig, HWKernelConfig, ModelSpecificConfig,
        SpeculativeExecutionConfig, CacheStoreConfig, MiscellaneousConfig,
        ArpcConfig, GrpcConfig,
    )
    from rtp_llm.config.kv_cache_config import KVCacheConfig
    from rtp_llm.config.py_config_modules import ServerConfig, LoadConfig
    from rtp_llm.config.engine_config import EngineConfig

    sc = ServerConfig()
    sc.start_port = -100
    kv = KVCacheConfig()
    kv.kv_cache_mem_mb = kv_cache_mb
    return EngineConfig(
        parallelism_config=ParallelismConfig(),
        runtime_config=RuntimeConfig(),
        nccl_comm_config=NcclCommConfig(),
        server_config=sc,
        pd_sep_config=PDSepConfig(),
        concurrency_config=ConcurrencyConfig(),
        fmha_config=FMHAConfig(),
        kv_cache_config=kv,
        profiling_debug_logging_config=ProfilingDebugLoggingConfig(),
        hw_kernel_config=HWKernelConfig(),
        device_resource_config=DeviceResourceConfig(),
        moe_config=MoeConfig(),
        model_specific_config=ModelSpecificConfig(),
        sp_config=SpeculativeExecutionConfig(),
        cache_store_config=CacheStoreConfig(),
        misc_config=MiscellaneousConfig(),
        arpc_config=ArpcConfig(),
        grpc_config=GrpcConfig(),
        load_config=LoadConfig(),
    )


def from_config(model_cls, config, engine_config, vit_config=None):
    kwargs = dict(
        model_config=config,
        parallelism_config=engine_config.parallelism_config,
        hw_kernel_config=engine_config.hw_kernel_config,
        kv_cache_config=engine_config.kv_cache_config,
        fmha_config=engine_config.fmha_config,
        moe_config=engine_config.moe_config,
        load_method=engine_config.load_config.load_method,
        max_generate_batch_size=engine_config.runtime_config.max_generate_batch_size,
        vit_config=vit_config,
        merge_lora=False,
        device_resource_config=engine_config.device_resource_config,
        force_cpu_load_weights=engine_config.load_config.force_cpu_load_weights,
    )
    sig = inspect.signature(model_cls.from_config)
    if 'load_python_model' in sig.parameters:
        kwargs['load_python_model'] = True
    if 'skip_python_model' in sig.parameters:
        kwargs['skip_python_model'] = False
    return model_cls.from_config(**kwargs)


def save_wav(waveform, path, sample_rate=24000):
    audio = waveform.squeeze().detach().cpu().float().numpy()
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    with open(path, "wb") as f:
        n = len(audio_int16)
        data_size = n * 2
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * 2))
        f.write(struct.pack("<H", 2))
        f.write(struct.pack("<H", 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(audio_int16.tobytes())


def wav_rms(waveform):
    """Root-mean-square of a waveform tensor — energy proxy."""
    a = waveform.squeeze().detach().cpu().float().numpy()
    return float(np.sqrt(np.mean(a * a)))


def main():
    from rtp_llm.omni.models.qwen2_5_omni.thinker import Qwen2_5OmniThinker
    from rtp_llm.omni.models.qwen2_5_omni.talker import Qwen2_5OmniTalker
    from rtp_llm.omni.models.qwen2_5_omni.token2wav_model import Token2WavModel
    from rtp_llm.config.py_config_modules import VitConfig
    from rtp_llm.async_decoder_engine.engine_creator import create_engine
    from transformers import AutoTokenizer

    logger.info("=" * 70)
    logger.info("Audio validation: REAL vs RANDOM thinker hidden states")
    logger.info("=" * 70)

    # ============== STAGE 1: thinker (to capture real hidden states) ==============
    logger.info("\n=== Loading thinker ===")
    eng_thinker = make_engine_config(kv_cache_mb=4096)
    tk_cfg = Qwen2_5OmniThinker._create_config(CKPT)
    tk_cfg.ckpt_path = CKPT
    tk_cfg.tokenizer_path = CKPT
    tk_cfg.model_type = "qwen2_5_omni_thinker"
    tk_cfg.max_seq_len = 4096
    tk_cfg.use_kvcache = True
    tk_cfg.phy2log_path = ""
    tk_cfg.init_precision_config(kv_cache_config=eng_thinker.kv_cache_config, act_type=None)
    tk_model = from_config(Qwen2_5OmniThinker, tk_cfg, eng_thinker, vit_config=VitConfig())
    tk_engine = create_engine(
        model=tk_model, engine_config=eng_thinker,
        alog_conf_path=eng_thinker.profiling_debug_logging_config.ft_alog_conf_path,
        world_info=None,
    )
    tk_engine.start()

    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    messages = [{"role": "user", "content": PROMPT}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt_text)
    input_ids = torch.tensor(prompt_ids, dtype=torch.int32)
    eos_token_id = tokenizer.eos_token_id or 151643

    logger.info("=== Thinker generate ===")
    output_tokens, thinker_hs_real_t = tk_engine.rtp_llm_op_.generate(
        input_ids, max_new_tokens=32, eos_token_id=eos_token_id, return_hidden_states=True,
    )
    num_gen = output_tokens.shape[1] if output_tokens.numel() > 0 else 0
    gen_text = tokenizer.decode(output_tokens[0].tolist(), skip_special_tokens=True)
    logger.info("Generated text: %r", gen_text[:200])
    logger.info("Real hidden states: shape=%s", tuple(thinker_hs_real_t.shape))

    # Move hidden states to CPU and free thinker engine
    real_hs_cpu = thinker_hs_real_t.cpu().clone()
    rand_hs_cpu = torch.randn_like(real_hs_cpu)

    logger.info("\n=== Stopping thinker engine ===")
    tk_engine.stop()
    del tk_engine, tk_model
    gc.collect()
    torch.cuda.empty_cache()
    free_mb, _ = torch.cuda.mem_get_info()
    logger.info("GPU free after thinker stop: %.1fGB", free_mb / 1024**3)

    # ============== STAGE 2: talker (used twice) ==============
    logger.info("\n=== Loading talker ===")
    eng_talker = make_engine_config(kv_cache_mb=2048)
    ta_cfg = Qwen2_5OmniTalker._create_config(CKPT)
    ta_cfg.ckpt_path = CKPT
    ta_cfg.tokenizer_path = CKPT
    ta_cfg.model_type = "qwen2_5_omni_talker"
    ta_cfg.max_seq_len = 2048
    ta_cfg.use_kvcache = True
    ta_cfg.phy2log_path = ""
    ta_cfg.init_precision_config(kv_cache_config=eng_talker.kv_cache_config, act_type=None)
    ta_model = from_config(Qwen2_5OmniTalker, ta_cfg, eng_talker)
    ta_engine = create_engine(
        model=ta_model, engine_config=eng_talker,
        alog_conf_path=eng_talker.profiling_debug_logging_config.ft_alog_conf_path,
        world_info=None,
    )
    ta_engine.start()

    py_model = ta_model.py_model
    device = "cuda:0"
    dtype = torch.bfloat16

    def run_talker(hidden_states_cpu, label):
        hs = hidden_states_cpu.to(device=device, dtype=dtype)
        py_model.set_thinker_hidden_states(hs)
        initial = torch.tensor([TALKER_CODEC_BOS], dtype=torch.int32)
        t0 = time.time()
        codec = ta_engine.rtp_llm_op_.generate(
            initial, max_new_tokens=200, eos_token_id=TALKER_CODEC_EOS,
        )
        py_model.clear_thinker_hidden_states()
        n = codec.shape[1] if codec.numel() > 0 else 0
        logger.info("[%s] talker generated %d codec tokens in %.2fs", label, n, time.time() - t0)
        return codec

    codec_real = run_talker(real_hs_cpu, "REAL")
    codec_rand = run_talker(rand_hs_cpu, "RAND")

    # Compare codec token sequences
    n_real = codec_real.shape[1]
    n_rand = codec_rand.shape[1]
    min_n = min(n_real, n_rand)
    if min_n == 0:
        logger.error("One of the runs produced 0 codec tokens — invalid comparison")
        return 1
    agree = (codec_real[0, :min_n] == codec_rand[0, :min_n]).float().mean().item()
    logger.info("Codec token agreement (first %d positions): %.1f%%", min_n, agree * 100)
    logger.info("Codec lengths: REAL=%d, RAND=%d", n_real, n_rand)

    # Filter special tokens, save WAVs
    talker_hs_shape = tuple(real_hs_cpu.shape)  # capture before del
    ta_engine.stop()
    del ta_engine, ta_model, py_model
    gc.collect()
    torch.cuda.empty_cache()
    free_mb, _ = torch.cuda.mem_get_info()
    logger.info("GPU free after talker stop: %.1fGB", free_mb / 1024**3)

    audio_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    token2wav = Token2WavModel.from_pretrained(CKPT, device=audio_device)
    spk_dict = torch.load(os.path.join(CKPT, "spk_dict.pt"), map_location=audio_device)
    spk = next(iter(spk_dict.values()))
    cond = spk["cond"].float().to(audio_device)
    ref_mel = spk["ref_mel"].float().to(audio_device)

    def codec_to_wav(codec_tokens, path):
        mask = codec_tokens[0] < 8292
        codec_filtered = codec_tokens[0][mask].unsqueeze(0).to(audio_device)
        if codec_filtered.shape[1] == 0:
            logger.error("All codec tokens were special for %s", path)
            return None
        with torch.no_grad():
            waveform = token2wav(codec_filtered, conditioning=cond, reference_mel=ref_mel)
        wav_cpu = waveform.cpu()
        del waveform, codec_filtered
        torch.cuda.empty_cache()
        save_wav(wav_cpu, path)
        rms = wav_rms(wav_cpu)
        logger.info("Saved %s: %.2fs RMS=%.4f",
                    path, wav_cpu.numel() / 24000, rms)
        return wav_cpu

    # Save only the REAL wav. The codec divergence above is the real validation;
    # the WAV is for manual A/B confirmation. Saving both runs the token2wav model
    # twice and can OOM on a single GPU.
    wav_real = codec_to_wav(codec_real, OUT_REAL)
    if wav_real is None:
        logger.error("FAIL: REAL WAV failed to generate")
        return 1
    rms_real = wav_rms(wav_real)
    if rms_real < 1e-3:
        logger.error("FAIL: REAL audio is silent (rms=%.5f)", rms_real)
        return 1

    # Validation: codec sequences must diverge meaningfully
    logger.info("\n" + "=" * 70)
    if agree > MAX_AGREEMENT_RATIO:
        logger.error(
            "FAIL: codec agreement %.1f%% exceeds threshold %.1f%% — "
            "talker may be ignoring hidden state conditioning",
            agree * 100, MAX_AGREEMENT_RATIO * 100,
        )
        return 1

    logger.info("PASS: real vs random hidden states produce divergent codec tokens")
    logger.info("  Generated text: %r", gen_text[:80])
    logger.info("  Thinker hidden_states: %s", talker_hs_shape)
    logger.info("  Codec agreement: %.1f%% (threshold <%.0f%%)", agree * 100, MAX_AGREEMENT_RATIO * 100)
    logger.info("  REAL wav (text speech): %s, RMS=%.4f", OUT_REAL, rms_real)
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
