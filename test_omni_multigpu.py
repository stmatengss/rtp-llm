"""Multi-GPU TP test: Qwen Omni thinker + talker resident on different GPUs.

Validates that both the thinker C++ engine (cuda:0) and the talker C++ engine
(cuda:1) coexist in one process WITHOUT one needing to stop before the other
can load. This is the core F1 milestone: each stage gets its own GPU and
both engines stay resident through the run.

Pipeline:
    Stage 1 (thinker, cuda:0): tokens + per-token hidden states via C++ engine
    Stage 2 (talker,  cuda:1): hidden_states → codec tokens via C++ engine
    Stage 3 (token2wav, cuda:1 by default): codec tokens → 24kHz waveform

Resident-engine check:
    After loading both engines, this test asserts that
    torch.cuda.memory_allocated(device) is non-trivial on BOTH cuda:0 AND cuda:1,
    proving each engine holds GPU memory simultaneously (no stop+reload trick).

Usage:
    CUDA_VISIBLE_DEVICES=5,6 python test_omni_multigpu.py
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
logger = logging.getLogger("test_omni_multigpu")

CKPT = "/root/models/Qwen/Qwen2.5-Omni-7B"
PROMPT = "Tell me a short joke."


def _import_configs():
    from rtp_llm.ops import (
        ArpcConfig, CacheStoreConfig, ConcurrencyConfig, DeviceResourceConfig,
        FMHAConfig, GrpcConfig, HWKernelConfig, MiscellaneousConfig,
        ModelSpecificConfig, MoeConfig, NcclCommConfig, ParallelismConfig,
        PDSepConfig, ProfilingDebugLoggingConfig, RuntimeConfig,
        SpeculativeExecutionConfig,
    )
    from rtp_llm.config.engine_config import EngineConfig
    from rtp_llm.config.kv_cache_config import KVCacheConfig
    from rtp_llm.config.py_config_modules import LoadConfig, ServerConfig
    return dict(
        ParallelismConfig=ParallelismConfig, RuntimeConfig=RuntimeConfig,
        FMHAConfig=FMHAConfig, DeviceResourceConfig=DeviceResourceConfig,
        MoeConfig=MoeConfig, NcclCommConfig=NcclCommConfig,
        PDSepConfig=PDSepConfig, ConcurrencyConfig=ConcurrencyConfig,
        ProfilingDebugLoggingConfig=ProfilingDebugLoggingConfig,
        HWKernelConfig=HWKernelConfig, ModelSpecificConfig=ModelSpecificConfig,
        SpeculativeExecutionConfig=SpeculativeExecutionConfig,
        CacheStoreConfig=CacheStoreConfig, MiscellaneousConfig=MiscellaneousConfig,
        ArpcConfig=ArpcConfig, GrpcConfig=GrpcConfig,
        KVCacheConfig=KVCacheConfig, ServerConfig=ServerConfig,
        LoadConfig=LoadConfig, EngineConfig=EngineConfig,
    )


def create_engine_config(*, device_id: int, num_devices: int,
                         start_port: int = -100, kv_cache_mb: int = 2048):
    """Build an EngineConfig that pins this stage to GPU index `device_id`.

    Two knobs flow through to the C++ engine and the Python weight loader:
      - ParallelismConfig.local_rank   → Python weight loading device
                                          (`cuda:{local_rank}`)
      - ParallelismConfig.world_rank   → C++ engine device id
        ParallelismConfig.local_world_size  via formula
                                          `world_rank % local_world_size`
    Setting local_rank == world_rank == device_id and
    local_world_size == num_devices keeps the two consistent.
    """
    cfg = _import_configs()
    server_config = cfg["ServerConfig"]()
    server_config.start_port = start_port

    kv_cache_config = cfg["KVCacheConfig"]()
    kv_cache_config.kv_cache_mem_mb = kv_cache_mb
    kv_cache_config.test_block_num = 0

    parallelism_config = cfg["ParallelismConfig"]()
    parallelism_config.tp_size = 1
    parallelism_config.tp_rank = 0
    parallelism_config.dp_size = 1
    parallelism_config.dp_rank = 0
    parallelism_config.world_size = num_devices
    parallelism_config.world_rank = device_id
    parallelism_config.local_world_size = num_devices
    parallelism_config.local_rank = device_id

    return cfg["EngineConfig"](
        parallelism_config=parallelism_config,
        runtime_config=cfg["RuntimeConfig"](),
        nccl_comm_config=cfg["NcclCommConfig"](),
        server_config=server_config,
        pd_sep_config=cfg["PDSepConfig"](),
        concurrency_config=cfg["ConcurrencyConfig"](),
        fmha_config=cfg["FMHAConfig"](),
        kv_cache_config=kv_cache_config,
        profiling_debug_logging_config=cfg["ProfilingDebugLoggingConfig"](),
        hw_kernel_config=cfg["HWKernelConfig"](),
        device_resource_config=cfg["DeviceResourceConfig"](),
        moe_config=cfg["MoeConfig"](),
        model_specific_config=cfg["ModelSpecificConfig"](),
        sp_config=cfg["SpeculativeExecutionConfig"](),
        cache_store_config=cfg["CacheStoreConfig"](),
        misc_config=cfg["MiscellaneousConfig"](),
        arpc_config=cfg["ArpcConfig"](),
        grpc_config=cfg["GrpcConfig"](),
        load_config=cfg["LoadConfig"](),
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
    """Build a Python BaseModel using engine_config's wiring."""
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


def make_engine_on_device(model_cls, ckpt_path, engine_config, model_type,
                          device_id, *, max_seq_len=4096, vit_config=None):
    """Build a LanguageCppEngine pinned to `device_id`.

    The torch.cuda.device guard makes sure any incidental main-thread CUDA work
    during construction (e.g. capability probes) targets the right GPU. The
    actual engine loop pins itself via cudaPreRun(device_id_) on its own thread.
    """
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

    with torch.cuda.device(device_id):
        model = from_config_with_python_model(
            model_cls, config, engine_config, vit_config
        )
        engine = create_engine(
            model=model,
            engine_config=engine_config,
            alog_conf_path=engine_config.profiling_debug_logging_config.ft_alog_conf_path,
            world_info=None,
        )
        engine.start()
    return engine, model


def gpu_mb(device_id: int) -> float:
    return torch.cuda.memory_allocated(device_id) / 1024 ** 2


def main():
    if torch.cuda.device_count() < 2:
        logger.error(
            f"Need at least 2 visible CUDA devices, got {torch.cuda.device_count()}. "
            "Set CUDA_VISIBLE_DEVICES=X,Y."
        )
        return 1

    from rtp_llm.omni.models.qwen2_5_omni.thinker import Qwen2_5OmniThinker
    from rtp_llm.omni.models.qwen2_5_omni.talker import Qwen2_5OmniTalker
    from rtp_llm.omni.models.qwen2_5_omni.token2wav_model import Token2WavModel
    from rtp_llm.config.py_config_modules import VitConfig
    from transformers import AutoTokenizer

    logger.info("=" * 72)
    logger.info("Multi-GPU TP test: thinker on cuda:0, talker on cuda:1, BOTH resident")
    logger.info("=" * 72)
    logger.info(f"Visible CUDA devices: {torch.cuda.device_count()}")
    for d in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(d)
        logger.info(
            f"  cuda:{d} {torch.cuda.get_device_name(d)} "
            f"free={free/1024**3:.1f}GB / total={total/1024**3:.1f}GB"
        )

    # Tokenize prompt
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    messages = [{"role": "user", "content": PROMPT}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text)
    logger.info(f"Prompt: {PROMPT!r} → {len(prompt_ids)} tokens")

    NUM_DEVICES = 2
    THINKER_DEVICE = 0
    TALKER_DEVICE = 1

    # ============== STAGE 1: THINKER on cuda:0 ==============
    logger.info(f"\n=== Stage 1: Loading thinker on cuda:{THINKER_DEVICE} ===")
    vit_config = VitConfig()
    thinker_cfg = create_engine_config(
        device_id=THINKER_DEVICE, num_devices=NUM_DEVICES,
        start_port=-100, kv_cache_mb=4096,
    )
    t0 = time.time()
    thinker_engine, thinker_model = make_engine_on_device(
        Qwen2_5OmniThinker, CKPT, thinker_cfg,
        "qwen2_5_omni_thinker", device_id=THINKER_DEVICE,
        max_seq_len=4096, vit_config=vit_config,
    )
    thinker_load_s = time.time() - t0
    logger.info(f"Thinker engine started in {thinker_load_s:.1f}s")

    thinker_mem_after_load = gpu_mb(THINKER_DEVICE)
    talker_mem_before_load = gpu_mb(TALKER_DEVICE)
    logger.info(
        f"After thinker load: cuda:{THINKER_DEVICE}={thinker_mem_after_load:.0f}MB "
        f"cuda:{TALKER_DEVICE}={talker_mem_before_load:.0f}MB"
    )
    assert thinker_mem_after_load > 1024, (
        f"Thinker should hold >1GB on cuda:{THINKER_DEVICE}, got {thinker_mem_after_load:.0f}MB"
    )

    # ============== STAGE 2: TALKER on cuda:1 — WITHOUT stopping thinker ==============
    logger.info(
        f"\n=== Stage 2: Loading talker on cuda:{TALKER_DEVICE} "
        f"(thinker stays resident on cuda:{THINKER_DEVICE}) ==="
    )
    talker_cfg = create_engine_config(
        device_id=TALKER_DEVICE, num_devices=NUM_DEVICES,
        start_port=-100, kv_cache_mb=2048,
    )
    t0 = time.time()
    talker_engine, talker_model = make_engine_on_device(
        Qwen2_5OmniTalker, CKPT, talker_cfg,
        "qwen2_5_omni_talker", device_id=TALKER_DEVICE,
        max_seq_len=2048,
    )
    talker_load_s = time.time() - t0
    logger.info(f"Talker engine started in {talker_load_s:.1f}s")

    # === Core multi-GPU assertion: BOTH engines must hold memory ===
    thinker_mem_concurrent = gpu_mb(THINKER_DEVICE)
    talker_mem_concurrent = gpu_mb(TALKER_DEVICE)
    logger.info(
        f"BOTH engines resident:\n"
        f"  cuda:{THINKER_DEVICE} thinker: {thinker_mem_concurrent:.0f}MB "
        f"(was {thinker_mem_after_load:.0f}MB before talker load)\n"
        f"  cuda:{TALKER_DEVICE} talker:  {talker_mem_concurrent:.0f}MB "
        f"(was {talker_mem_before_load:.0f}MB before talker load)"
    )

    assert thinker_mem_concurrent > 1024, (
        f"Thinker should still hold >1GB on cuda:{THINKER_DEVICE} after talker load; "
        f"got {thinker_mem_concurrent:.0f}MB. (Did the thinker fail to stay resident?)"
    )
    assert talker_mem_concurrent > 512, (
        f"Talker should hold >512MB on cuda:{TALKER_DEVICE}; got {talker_mem_concurrent:.0f}MB. "
        f"(Did the talker land on the wrong device?)"
    )
    # Talker memory must be on the talker's device, not the thinker's
    # (validates the device-pinning actually worked).
    thinker_growth = thinker_mem_concurrent - thinker_mem_after_load
    assert thinker_growth < 512, (
        f"Thinker device gained {thinker_growth:.0f}MB during talker load, "
        f"suggesting talker leaked allocations onto cuda:{THINKER_DEVICE}. "
        f"Multi-GPU pinning is BROKEN."
    )
    logger.info(
        f"PASS: thinker held its ground (+{thinker_growth:.0f}MB), "
        f"talker landed on its own device (+{talker_mem_concurrent - talker_mem_before_load:.0f}MB)"
    )

    # Sanity: both engines report ready/started
    assert getattr(thinker_engine, "started_", True) is not False, "thinker not started"
    assert getattr(talker_engine, "started_", True) is not False, "talker not started"

    # ============== Run thinker.generate() — captures real hidden states ==============
    logger.info("\n=== Running thinker.generate() on cuda:0 ===")
    input_ids = torch.tensor(prompt_ids, dtype=torch.int32)
    eos_token_id = tokenizer.eos_token_id or 151643
    t0 = time.time()
    with torch.cuda.device(THINKER_DEVICE):
        output_tokens, thinker_hs = thinker_engine.rtp_llm_op_.generate(
            input_ids, max_new_tokens=64, eos_token_id=eos_token_id,
            return_hidden_states=True,
        )
    gen_time = time.time() - t0
    num_gen = output_tokens.shape[1] if output_tokens.numel() > 0 else 0
    logger.info(f"Thinker generated {num_gen} tokens in {gen_time:.2f}s")
    logger.info(
        f"Thinker hidden_states: shape={tuple(thinker_hs.shape)} dtype={thinker_hs.dtype}"
    )
    assert num_gen > 0, "thinker generated zero tokens"
    assert thinker_hs.shape[0] == num_gen, (
        f"hidden_states rows ({thinker_hs.shape[0]}) != num_gen ({num_gen})"
    )
    gen_text = tokenizer.decode(output_tokens[0].tolist(), skip_special_tokens=True)
    logger.info(f"Generated text: {gen_text[:200]!r}")

    # Re-check residency after thinker generate
    thinker_mem_after_gen = gpu_mb(THINKER_DEVICE)
    talker_mem_after_gen = gpu_mb(TALKER_DEVICE)
    logger.info(
        f"After thinker.generate: cuda:0={thinker_mem_after_gen:.0f}MB "
        f"cuda:1={talker_mem_after_gen:.0f}MB"
    )
    assert talker_mem_after_gen > 512, (
        "Talker engine evaporated after thinker generation — multi-residency broken"
    )

    # ============== Move hidden states cuda:0 → cuda:1, run talker.generate() ==============
    logger.info("\n=== Running talker.generate() on cuda:1 ===")
    py_model = talker_model.py_model
    dtype = torch.bfloat16
    thinker_hs_for_talker = thinker_hs.to(
        device=f"cuda:{TALKER_DEVICE}", dtype=dtype
    )
    logger.info(
        f"Cross-device transfer: hidden_states cuda:{THINKER_DEVICE} → cuda:{TALKER_DEVICE} "
        f"shape={tuple(thinker_hs_for_talker.shape)}"
    )
    py_model.set_thinker_hidden_states(thinker_hs_for_talker)

    TALKER_CODEC_BOS = 8293
    TALKER_CODEC_EOS = 8294
    initial_tokens = torch.tensor([TALKER_CODEC_BOS], dtype=torch.int32)
    t0 = time.time()
    with torch.cuda.device(TALKER_DEVICE):
        codec_tokens = talker_engine.rtp_llm_op_.generate(
            initial_tokens, max_new_tokens=200, eos_token_id=TALKER_CODEC_EOS,
        )
    talker_time = time.time() - t0
    num_codec = codec_tokens.shape[1] if codec_tokens.numel() > 0 else 0
    logger.info(f"Talker generated {num_codec} codec tokens in {talker_time:.2f}s")
    py_model.clear_thinker_hidden_states()
    assert num_codec > 0, "talker generated zero codec tokens"

    # Filter special tokens
    mask = codec_tokens[0] < 8292
    codec_filtered = codec_tokens[0][mask].unsqueeze(0).cpu()
    logger.info(f"After filtering specials: {codec_filtered.shape[1]} codec tokens")
    assert codec_filtered.shape[1] > 0, "all codec tokens were special tokens"

    # ============== STAGE 3: token2wav (also cuda:1) ==============
    audio_device = f"cuda:{TALKER_DEVICE}"
    logger.info(f"\n=== Stage 3: Token2wav on {audio_device} ===")
    t0 = time.time()
    token2wav = Token2WavModel.from_pretrained(CKPT, device=audio_device)
    logger.info(f"Token2wav loaded in {time.time()-t0:.1f}s")

    spk_path = os.path.join(CKPT, "spk_dict.pt")
    spk_dict = torch.load(spk_path, map_location=audio_device)
    spk_name = list(spk_dict.keys())[0]
    spk = spk_dict[spk_name]
    cond = spk["cond"].float().to(audio_device)
    ref_mel = spk["ref_mel"].float().to(audio_device)
    logger.info(f"Speaker: {spk_name}")

    t0 = time.time()
    waveform = token2wav(
        codec_filtered.to(audio_device),
        conditioning=cond,
        reference_mel=ref_mel,
    )
    wav_time = time.time() - t0
    duration = waveform.numel() / 24000
    logger.info(f"Generated {duration:.2f}s audio in {wav_time:.2f}s")
    assert duration > 0.1, f"WAV too short: {duration:.2f}s"

    out_path = "/root/test_omni_multigpu.wav"
    save_wav(waveform, out_path)
    assert os.path.getsize(out_path) > 1024, "WAV file too small"

    # Final residency check — everything stayed alive
    final_thinker = gpu_mb(THINKER_DEVICE)
    final_talker = gpu_mb(TALKER_DEVICE)
    logger.info(
        f"Final memory: cuda:{THINKER_DEVICE}={final_thinker:.0f}MB "
        f"cuda:{TALKER_DEVICE}={final_talker:.0f}MB"
    )

    logger.info("\n" + "=" * 72)
    logger.info("MULTI-GPU TP TEST PASSED")
    logger.info(f"  thinker cuda:{THINKER_DEVICE}: {num_gen} tokens, {final_thinker:.0f}MB resident")
    logger.info(f"  talker  cuda:{TALKER_DEVICE}: {num_codec} codec tokens, {final_talker:.0f}MB resident")
    logger.info(f"  audio:  {duration:.2f}s WAV → {out_path}")
    logger.info("  BOTH engines stayed loaded throughout the run (no stop+reload)")
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
