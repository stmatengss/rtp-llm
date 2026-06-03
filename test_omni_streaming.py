"""F2: Interleaved streaming thinker→talker e2e test.

The talker engine starts generating codec tokens AS SOON AS the thinker emits
its first hidden state, rather than waiting for the thinker to reach EOS.
Measures `t_first_codec` vs `t_thinker_done`; if the streaming pipeline is
working, `t_first_codec < t_thinker_done`.

Architecture
============
- Both thinker and talker engines are loaded resident in the same process.
  The C++ runtime currently binds globally to a single device via
  ``g_device_id`` (set via ``initRuntime`` ``call_once``), so both engines
  share ``cuda:0`` on the same physical GPU. Multi-GPU residency is F1's job;
  this F2 test still demonstrates the streaming pipeline because the two
  engines progress concurrently via independent Python threads and the C++
  engines release the GIL between steps.
- A dedicated Python thread drives ``talker_engine.generate_with_callback``.
  Its first-step callback timestamps ``t_first_codec``.
- The main thread drives ``thinker_engine.generate_with_callback`` and the
  per-step callback pushes hidden states into the talker model so the talker's
  ``forward()`` (blocking on a threading.Event) can advance.

Usage:
    CUDA_VISIBLE_DEVICES=5 python test_omni_streaming.py     # single GPU
    CUDA_VISIBLE_DEVICES=5,6 python test_omni_streaming.py   # also works
"""
import gc
import logging
import os
import struct
import sys
import threading
import time
import inspect

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger("test_omni_streaming")

CKPT = "/root/models/Qwen/Qwen2.5-Omni-7B"
PROMPT = "Tell me a short joke."

TALKER_CODEC_BOS = 8293
TALKER_CODEC_EOS = 8294


def create_engine_config(start_port=-100, kv_cache_mb=2048):
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

    server_config = ServerConfig()
    server_config.start_port = start_port

    kv_cache_config = KVCacheConfig()
    kv_cache_config.kv_cache_mem_mb = kv_cache_mb
    kv_cache_config.test_block_num = 0

    return EngineConfig(
        parallelism_config=ParallelismConfig(),
        runtime_config=RuntimeConfig(),
        nccl_comm_config=NcclCommConfig(),
        server_config=server_config,
        pd_sep_config=PDSepConfig(),
        concurrency_config=ConcurrencyConfig(),
        fmha_config=FMHAConfig(),
        kv_cache_config=kv_cache_config,
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


def save_wav(waveform, path, sample_rate=24000):
    audio = waveform.squeeze().detach().cpu().float().numpy()
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    with open(path, "wb") as f:
        num_samples = len(audio_int16)
        data_size = num_samples * 2
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
    logger.info(f"WAV saved: {path} ({num_samples/sample_rate:.2f}s, {os.path.getsize(path)} bytes)")


def from_config_with_python_model(model_cls, config, engine_config, vit_config=None):
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


def make_engine(model_cls, ckpt_path, engine_config, model_type, max_seq_len=4096, vit_config=None):
    from rtp_llm.async_decoder_engine.engine_creator import create_engine

    config = model_cls._create_config(ckpt_path)
    config.ckpt_path = ckpt_path
    config.tokenizer_path = ckpt_path
    config.model_type = model_type
    config.max_seq_len = max_seq_len
    config.use_kvcache = True
    config.phy2log_path = ""
    config.init_precision_config(
        kv_cache_config=engine_config.kv_cache_config, act_type=None
    )

    model = from_config_with_python_model(model_cls, config, engine_config, vit_config)
    engine = create_engine(
        model=model,
        engine_config=engine_config,
        alog_conf_path=engine_config.profiling_debug_logging_config.ft_alog_conf_path,
        world_info=None,
    )
    engine.start()
    return engine, model


def main():
    from rtp_llm.omni.models.qwen2_5_omni.thinker import Qwen2_5OmniThinker
    from rtp_llm.omni.models.qwen2_5_omni.talker import Qwen2_5OmniTalker
    from rtp_llm.omni.models.qwen2_5_omni.token2wav_model import Token2WavModel
    from rtp_llm.config.py_config_modules import VitConfig
    from transformers import AutoTokenizer

    logger.info("=" * 70)
    logger.info("F2 streaming e2e: interleaved thinker→talker via callback API")
    logger.info("=" * 70)

    # Tokenize prompt
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    messages = [{"role": "user", "content": PROMPT}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt_text)
    logger.info(f"Prompt: {PROMPT!r} → {len(prompt_ids)} ids")

    num_devices = torch.cuda.device_count()
    logger.info(f"CUDA devices visible: {num_devices} (using cuda:0 for both engines)")

    # Aggressive KV-cache caps so both engines fit on one 22GB GPU.
    # 64MB ≈ a handful of blocks; enough for the short prompt + ≤64 generated
    # text tokens + ≤200 codec tokens used by this test.
    thinker_kv_mb = int(os.environ.get("THINKER_KV_MB", "256"))
    talker_kv_mb = int(os.environ.get("TALKER_KV_MB", "128"))

    # ============== Load thinker (small KV cache so talker fits too) ==============
    logger.info("\n=== Loading thinker via RTP engine ===")
    vit_config = VitConfig()
    engine_config_thinker = create_engine_config(start_port=-100, kv_cache_mb=thinker_kv_mb)
    t0 = time.time()
    thinker_engine, thinker_model = make_engine(
        Qwen2_5OmniThinker, CKPT, engine_config_thinker, "qwen2_5_omni_thinker",
        max_seq_len=2048, vit_config=vit_config,
    )
    logger.info(f"Thinker engine started in {time.time()-t0:.1f}s")
    assert hasattr(thinker_engine.rtp_llm_op_.ft_op, 'generate_with_callback'), (
        "C++ generate_with_callback() missing — rebuild RtpLLMOp via "
        "bazelisk build //:th_transformer"
    )

    free_mb, total_mb = torch.cuda.mem_get_info()
    logger.info(f"GPU free after thinker: {free_mb/1024**3:.1f}GB / {total_mb/1024**3:.1f}GB")

    # ============== Load talker resident on the same GPU ==============
    logger.info("\n=== Loading talker resident on same GPU ===")
    engine_config_talker = create_engine_config(start_port=-110, kv_cache_mb=talker_kv_mb)
    t0 = time.time()
    talker_engine, talker_model = make_engine(
        Qwen2_5OmniTalker, CKPT, engine_config_talker, "qwen2_5_omni_talker",
        max_seq_len=1024,
    )
    logger.info(f"Talker engine started in {time.time()-t0:.1f}s")
    free_mb, total_mb = torch.cuda.mem_get_info()
    logger.info(f"GPU free after both resident: {free_mb/1024**3:.1f}GB / {total_mb/1024**3:.1f}GB")

    talker_py = talker_model.py_model
    talker_py.begin_streaming_thinker()

    # ============== STREAMING ==============
    logger.info("\n=== Interleaved streaming generate_with_callback() ===")
    input_ids = torch.tensor(prompt_ids, dtype=torch.int32)
    eos_token_id = tokenizer.eos_token_id or 151643

    metrics = {
        "t_start": 0.0,
        "t_first_thinker_token": None,
        "t_first_hs_pushed": None,
        "t_thinker_done": None,
        "t_first_codec": None,
        "t_last_codec": None,
        "num_codec": 0,
    }
    thinker_steps = {"count": 0}
    codec_steps = {"count": 0}

    def thinker_cb(tokens, hidden_states, finished):
        now = time.time() - metrics["t_start"]
        if tokens is not None and tokens.numel() > 0:
            thinker_steps["count"] += int(tokens.numel())
            if metrics["t_first_thinker_token"] is None:
                metrics["t_first_thinker_token"] = now
        if hidden_states is not None and hidden_states.numel() > 0:
            if metrics["t_first_hs_pushed"] is None:
                metrics["t_first_hs_pushed"] = now
            talker_py.push_thinker_hidden_state(hidden_states)
        if finished:
            metrics["t_thinker_done"] = now
            talker_py.mark_thinker_done()

    def talker_cb(tokens, _hs, _finished):
        now = time.time() - metrics["t_start"]
        if tokens is not None and tokens.numel() > 0:
            codec_steps["count"] += int(tokens.numel())
            if metrics["t_first_codec"] is None:
                metrics["t_first_codec"] = now
            metrics["t_last_codec"] = now

    talker_result_box = {"tokens": None, "exc": None}

    def talker_thread_fn():
        try:
            initial = torch.tensor([TALKER_CODEC_BOS], dtype=torch.int32)
            tokens, _ = talker_engine.rtp_llm_op_.generate_with_callback(
                initial, talker_cb,
                max_new_tokens=200, eos_token_id=TALKER_CODEC_EOS,
            )
            talker_result_box["tokens"] = tokens
        except Exception as e:  # noqa: BLE001
            talker_result_box["exc"] = e
            logger.exception("talker thread failed")

    talker_thread = threading.Thread(target=talker_thread_fn, name="talker-stream")

    metrics["t_start"] = time.time()
    talker_thread.start()

    output_tokens, thinker_hs_full = thinker_engine.rtp_llm_op_.generate_with_callback(
        input_ids, thinker_cb,
        max_new_tokens=64, eos_token_id=eos_token_id,
    )
    if metrics["t_thinker_done"] is None:
        metrics["t_thinker_done"] = time.time() - metrics["t_start"]
        talker_py.mark_thinker_done()

    num_gen = output_tokens.shape[1] if output_tokens.numel() > 0 else 0
    logger.info(
        f"Thinker: {num_gen} tokens in {metrics['t_thinker_done']:.2f}s "
        f"(first token at {metrics['t_first_thinker_token']}s, "
        f"first hs at {metrics['t_first_hs_pushed']}s)"
    )
    if num_gen > 0:
        gen_text = tokenizer.decode(output_tokens[0].tolist(), skip_special_tokens=True)
        logger.info(f"Generated text: {gen_text[:200]!r}")

    talker_thread.join(timeout=180)
    if talker_thread.is_alive():
        raise RuntimeError("Talker stream thread did not finish within 180s")
    if talker_result_box["exc"] is not None:
        raise talker_result_box["exc"]

    codec_tokens = talker_result_box["tokens"]
    metrics["num_codec"] = (
        codec_tokens.shape[1] if codec_tokens is not None and codec_tokens.numel() > 0 else 0
    )
    logger.info(
        f"Talker: {metrics['num_codec']} codec tokens; "
        f"first at {metrics['t_first_codec']}s, last at {metrics['t_last_codec']}s"
    )

    # ============== Validation ==============
    assert metrics["num_codec"] > 0, "no codec tokens generated"
    assert metrics["t_first_codec"] is not None, "first-codec timestamp missing"
    assert metrics["t_thinker_done"] is not None, "thinker-done timestamp missing"

    # The interleaving claim: first codec token arrived BEFORE thinker finished.
    margin = metrics["t_thinker_done"] - metrics["t_first_codec"]
    logger.info(
        f"\nInterleaving margin: t_thinker_done - t_first_codec = "
        f"{metrics['t_thinker_done']:.3f}s - {metrics['t_first_codec']:.3f}s = "
        f"{margin:.3f}s"
    )
    assert metrics["t_first_codec"] < metrics["t_thinker_done"], (
        f"Streaming pipeline failed: first codec token came AFTER thinker "
        f"finished (first_codec={metrics['t_first_codec']:.3f}s, "
        f"thinker_done={metrics['t_thinker_done']:.3f}s)"
    )

    # Filter codec tokens
    mask = codec_tokens[0] < 8292
    codec_filtered = codec_tokens[0][mask].unsqueeze(0).cpu()
    logger.info(f"After filtering specials: {codec_filtered.shape[1]} valid codec tokens")

    # Persist the codec tokens + metrics BEFORE attempting any engine teardown.
    # Engine.stop() under this much memory pressure tends to throw std::bad_alloc
    # from a background thread, which std::terminate()s the process. So we
    # save what we need for the audio validation now, then either:
    #   (a) inline token2wav succeeds (memory allowing), or
    #   (b) we hard-exit (os._exit) past the engine destructors and a
    #       follow-up subprocess validates the WAV.
    codec_path = "/tmp/test_omni_streaming.codec.pt"
    torch.save(codec_filtered, codec_path)
    logger.info(f"Wrote codec tokens to {codec_path}")

    out_path = "/root/test_omni_streaming.wav"
    audio_ok = False
    try:
        run_audio_validation_inline(codec_filtered, out_path)
        audio_ok = True
    except Exception as e:  # noqa: BLE001
        logger.warning("Inline token2wav failed (%s); will validate via subprocess", e)

    if not audio_ok:
        # Fork a subprocess that knows nothing about our two engines and
        # so won't fight them for GPU memory. We hard-exit afterwards to
        # skip Python-level engine cleanup (which is the source of the
        # std::bad_alloc crash under this memory pressure).
        #
        # The parent has saturated whichever physical GPU is exposed as cuda:0.
        # Hand the subprocess a different physical device via CUDA_VISIBLE_DEVICES
        # so it can actually allocate. Requires at least 2 physical GPUs visible
        # (typical: run as ``CUDA_VISIBLE_DEVICES=5,6`` or ``=5,6,7,4``).
        import subprocess
        env = os.environ.copy()
        cvd_in = env.get("CUDA_VISIBLE_DEVICES", "")
        cvd_list = [x for x in cvd_in.split(",") if x.strip()]
        if len(cvd_list) >= 2:
            env["CUDA_VISIBLE_DEVICES"] = cvd_list[1]  # second physical device
        else:
            logger.warning(
                "Only %d device(s) visible (CUDA_VISIBLE_DEVICES=%r); "
                "audio subprocess will compete for the same GPU and likely OOM. "
                "Pass a second GPU id to fix.",
                len(cvd_list), cvd_in,
            )
        cmd = [
            sys.executable, __file__, "--audio-only",
            "--codec", codec_path, "--wav", out_path,
        ]
        logger.info(
            f"Spawning audio-validation subprocess (CUDA_VISIBLE_DEVICES="
            f"{env.get('CUDA_VISIBLE_DEVICES')!r}): {' '.join(cmd)}"
        )
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            logger.error("Audio validation subprocess failed (rc=%d)", proc.returncode)
            # Hard-exit so the streaming portion's success is preserved as a
            # non-zero exit code (we still want the user to know audio failed).
            os._exit(2)
        audio_ok = True

    logger.info("\n" + "=" * 70)
    logger.info("F2 STREAMING TEST PASSED")
    logger.info(f"  Thinker:        {num_gen} tokens in {metrics['t_thinker_done']:.2f}s")
    logger.info(f"  First HS push:  {metrics['t_first_hs_pushed']:.3f}s")
    logger.info(f"  First codec:    {metrics['t_first_codec']:.3f}s "
                f"(margin: {margin:.3f}s before thinker finished)")
    logger.info(f"  Last codec:     {metrics['t_last_codec']:.3f}s")
    logger.info(f"  Codec tokens:   {metrics['num_codec']} → {codec_filtered.shape[1]} after filter")
    logger.info(f"  Audio:          validated → {out_path}")
    logger.info("=" * 70)
    # Skip Python-level engine destructors — they bad_alloc under this memory
    # pressure. The OS reclaims everything on process exit.
    os._exit(0)


def run_audio_validation_inline(codec_filtered, out_path):
    """Try token2wav inline. Raises on OOM/other errors."""
    from rtp_llm.omni.models.qwen2_5_omni.token2wav_model import Token2WavModel
    audio_device = os.environ.get(
        "AUDIO_DEVICE", "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    )
    logger.info(f"Token2wav (inline, device={audio_device})...")
    token2wav = Token2WavModel.from_pretrained(CKPT, device=audio_device)
    spk_path = os.path.join(CKPT, "spk_dict.pt")
    spk_dict = torch.load(spk_path, map_location=audio_device)
    spk_name = list(spk_dict.keys())[0]
    spk = spk_dict[spk_name]
    cond = spk["cond"].float().to(audio_device)
    ref_mel = spk["ref_mel"].float().to(audio_device)
    waveform = token2wav(
        codec_filtered.to(audio_device), conditioning=cond, reference_mel=ref_mel,
    )
    _validate_and_save(waveform, out_path)


def _validate_and_save(waveform, out_path):
    duration = waveform.numel() / 24000
    logger.info(f"Generated {duration:.2f}s audio")
    assert waveform.numel() > 0, "empty waveform"
    audio_np = waveform.detach().float().cpu().numpy()
    assert np.isfinite(audio_np).all(), "waveform contains NaN/Inf"
    assert float(np.abs(audio_np).max()) > 1e-4, "waveform is silent / all-zeros"
    save_wav(waveform, out_path)


def audio_only_main(codec_path, wav_path):
    """Subprocess entry: load codec tokens from disk, run token2wav, save WAV."""
    from rtp_llm.omni.models.qwen2_5_omni.token2wav_model import Token2WavModel
    logger.info(f"[audio-only] Loading codec tokens from {codec_path}")
    codec_filtered = torch.load(codec_path, map_location="cpu")
    audio_device = os.environ.get("AUDIO_DEVICE", "cuda:0")
    logger.info(f"[audio-only] Token2wav on {audio_device}")
    token2wav = Token2WavModel.from_pretrained(CKPT, device=audio_device)
    spk_path = os.path.join(CKPT, "spk_dict.pt")
    spk_dict = torch.load(spk_path, map_location=audio_device)
    spk_name = list(spk_dict.keys())[0]
    spk = spk_dict[spk_name]
    cond = spk["cond"].float().to(audio_device)
    ref_mel = spk["ref_mel"].float().to(audio_device)
    waveform = token2wav(
        codec_filtered.to(audio_device), conditioning=cond, reference_mel=ref_mel,
    )
    _validate_and_save(waveform, wav_path)
    logger.info(f"[audio-only] WAV validated: {wav_path}")
    return 0


if __name__ == "__main__":
    if "--audio-only" in sys.argv:
        i_codec = sys.argv.index("--codec") + 1
        i_wav = sys.argv.index("--wav") + 1
        sys.exit(audio_only_main(sys.argv[i_codec], sys.argv[i_wav]))
    sys.exit(main())
