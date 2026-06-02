"""Test talker generation via new C++ engine generate() with thinker hidden state injection.

Validates:
  1. Qwen2_5OmniTalker loads with python model (Qwen2_5OmniTalkerModel)
  2. C++ engine's generate() method works (rtp_llm_op_.generate)
  3. Python model receives thinker hidden states via set_thinker_hidden_states()
  4. Codec tokens are produced and convertible to audio via token2wav

Usage:
    CUDA_VISIBLE_DEVICES=5 python test_talker_cpp_engine.py
"""
import logging
import os
import struct
import sys
import time

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger("test_talker_cpp_engine")

CKPT = "/root/models/Qwen/Qwen2.5-Omni-7B"


def create_engine_config(disable_server=True):
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
    if disable_server:
        server_config.start_port = -100  # negative → skip gRPC bind

    return EngineConfig(
        parallelism_config=ParallelismConfig(),
        runtime_config=RuntimeConfig(),
        nccl_comm_config=NcclCommConfig(),
        server_config=server_config,
        pd_sep_config=PDSepConfig(),
        concurrency_config=ConcurrencyConfig(),
        fmha_config=FMHAConfig(),
        kv_cache_config=KVCacheConfig(),
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
    logger.info(f"WAV saved to {path} ({num_samples/sample_rate:.2f}s)")


def main():
    from rtp_llm.omni.models.qwen2_5_omni.talker import Qwen2_5OmniTalker
    from rtp_llm.omni.models.qwen2_5_omni.token2wav_model import Token2WavModel
    from rtp_llm.async_decoder_engine.engine_creator import create_engine
    import inspect

    logger.info("=== Test: Talker via C++ engine generate() ===")
    engine_config = create_engine_config()

    # Load talker
    talker_config = Qwen2_5OmniTalker._create_config(CKPT)
    talker_config.ckpt_path = CKPT
    talker_config.tokenizer_path = CKPT
    talker_config.model_type = "qwen2_5_omni_talker"
    talker_config.max_seq_len = 2048
    talker_config.use_kvcache = True
    talker_config.phy2log_path = ""
    talker_config.init_precision_config(
        kv_cache_config=engine_config.kv_cache_config, act_type=None
    )

    from_config_kwargs = dict(
        model_config=talker_config,
        parallelism_config=engine_config.parallelism_config,
        hw_kernel_config=engine_config.hw_kernel_config,
        kv_cache_config=engine_config.kv_cache_config,
        fmha_config=engine_config.fmha_config,
        moe_config=engine_config.moe_config,
        load_method=engine_config.load_config.load_method,
        max_generate_batch_size=engine_config.runtime_config.max_generate_batch_size,
        vit_config=None,
        merge_lora=False,
        device_resource_config=engine_config.device_resource_config,
        force_cpu_load_weights=engine_config.load_config.force_cpu_load_weights,
    )
    sig = inspect.signature(Qwen2_5OmniTalker.from_config)
    if 'skip_python_model' in sig.parameters:
        from_config_kwargs['skip_python_model'] = False

    t0 = time.time()
    talker_model = Qwen2_5OmniTalker.from_config(**from_config_kwargs)
    logger.info(f"Talker loaded in {time.time()-t0:.1f}s")
    logger.info(f"py_model: {type(talker_model.py_model).__name__}")

    # Create and start engine
    alog_conf_path = engine_config.profiling_debug_logging_config.ft_alog_conf_path
    t0 = time.time()
    talker_engine = create_engine(
        model=talker_model,
        engine_config=engine_config,
        alog_conf_path=alog_conf_path,
        world_info=None,
    )
    talker_engine.start()
    logger.info(f"Talker engine started in {time.time()-t0:.1f}s")

    # Verify the C++ generate() is available
    op = talker_engine.rtp_llm_op_
    if not hasattr(op.ft_op, 'generate'):
        logger.error("C++ generate() not available — build may be stale")
        return 1

    # Load token2wav
    device = "cuda:0"
    t0 = time.time()
    token2wav = Token2WavModel.from_pretrained(CKPT, device=device)
    logger.info(f"Token2wav loaded in {time.time()-t0:.1f}s")

    # Load speaker data
    spk_path = os.path.join(CKPT, "spk_dict.pt")
    spk_dict = torch.load(spk_path, map_location=device)
    speaker_name = list(spk_dict.keys())[0]
    spk = spk_dict[speaker_name]
    spk_bos = spk["bos_token"]
    cond = spk["cond"].float().to(device)
    ref_mel = spk["ref_mel"].float().to(device)
    logger.info(f"Speaker: {speaker_name}, BOS: {spk_bos}")

    # Set up thinker hidden states (use random for this test)
    py_model = talker_model.py_model
    num_thinker_tokens = 60
    dtype = torch.bfloat16
    thinker_hs = torch.randn(num_thinker_tokens, 3584, dtype=dtype, device=device)
    logger.info(f"Thinker hidden states shape: {thinker_hs.shape}")

    py_model.set_thinker_hidden_states(thinker_hs)

    # Generate codec tokens via C++ engine
    initial_tokens = torch.tensor([spk_bos], dtype=torch.int32)
    eos_token_id = 8294  # talker EOS

    logger.info("=== Generating codec tokens via C++ engine ===")
    t0 = time.time()
    codec_tokens = op.generate(initial_tokens, max_new_tokens=200, eos_token_id=eos_token_id)
    gen_time = time.time() - t0
    num_codec = codec_tokens.shape[1] if codec_tokens.numel() > 0 else 0
    logger.info(f"Generated {num_codec} codec tokens in {gen_time:.2f}s")

    py_model.clear_thinker_hidden_states()

    if num_codec == 0:
        logger.error("No codec tokens generated!")
        return 1

    logger.info(f"Codec tokens (first 30): {codec_tokens[0, :30].tolist()}")

    # Filter special tokens
    mask = codec_tokens[0] < 8292
    codec_filtered = codec_tokens[0][mask].unsqueeze(0)
    logger.info(f"After filtering: {codec_filtered.shape[1]} valid codec tokens")

    if codec_filtered.shape[1] == 0:
        logger.error("All codec tokens were special tokens!")
        return 1

    # Generate audio
    t0 = time.time()
    waveform = token2wav(
        codec_filtered.to(device),
        conditioning=cond,
        reference_mel=ref_mel,
    )
    wav_time = time.time() - t0
    duration = waveform.numel() / 24000
    logger.info(f"Generated {duration:.2f}s audio in {wav_time:.2f}s")

    save_wav(waveform, "/root/test_talker_cpp_engine.wav")
    logger.info("Talker C++ engine e2e test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
