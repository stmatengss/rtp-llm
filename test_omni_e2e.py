"""End-to-end test: OmniEngine loads thinker + talker, generates audio.

Usage:
    CUDA_VISIBLE_DEVICES=5 python test_omni_e2e.py
"""
import logging
import os
import struct
import time

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger("test_omni_e2e")

CKPT = "/root/models/Qwen/Qwen2.5-Omni-7B"


def create_engine_config():
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

    return EngineConfig(
        parallelism_config=ParallelismConfig(),
        runtime_config=RuntimeConfig(),
        nccl_comm_config=NcclCommConfig(),
        server_config=ServerConfig(),
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


def test_talker_only_e2e():
    """Test talker engine + token2wav without thinker (use random hidden states)."""
    from rtp_llm.omni.models.qwen2_5_omni.talker import Qwen2_5OmniTalker
    from rtp_llm.omni.models.qwen2_5_omni.talker_engine_wrapper import TalkerEngineWrapper
    from rtp_llm.omni.models.qwen2_5_omni.token2wav_model import Token2WavModel
    import inspect

    logger.info("=== Test: talker engine → codec tokens → token2wav → WAV ===")

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
    if 'load_python_model' in sig.parameters:
        from_config_kwargs['load_python_model'] = True
    if 'skip_python_model' in sig.parameters:
        from_config_kwargs['skip_python_model'] = False

    t0 = time.time()
    talker_model = Qwen2_5OmniTalker.from_config(**from_config_kwargs)
    logger.info(f"Talker loaded in {time.time()-t0:.1f}s")

    class FakeEngine:
        def __init__(self, model):
            self.model = model

    wrapper = TalkerEngineWrapper(FakeEngine(talker_model))

    device = wrapper.proj_weight.device
    dtype = wrapper.dtype

    # Simulate thinker hidden states (50 tokens worth)
    num_tokens = 50
    thinker_hs = torch.randn(num_tokens, 3584, dtype=dtype, device=device)

    # Speaker BOS token
    initial_tokens = torch.tensor([8293], dtype=torch.long, device=device)

    # Use -1 as eos_token_id to prevent early stopping with random hidden states
    t0 = time.time()
    codec_tokens = wrapper.generate(
        thinker_hidden_states=thinker_hs,
        initial_token_ids=initial_tokens,
        max_new_tokens=200,
        eos_token_id=-1,
    )
    gen_time = time.time() - t0
    logger.info(f"Generated {codec_tokens.shape[1]} codec tokens in {gen_time:.2f}s")

    # Filter special tokens
    mask = codec_tokens[0] < 8292
    codec_filtered = codec_tokens[0][mask].unsqueeze(0)
    logger.info(f"After filtering: {codec_filtered.shape[1]} valid codec tokens")

    if codec_filtered.shape[1] == 0:
        logger.warning("No valid codec tokens — skipping token2wav")
        return

    # Load token2wav
    t0 = time.time()
    token2wav = Token2WavModel.from_pretrained(CKPT, device=str(device))
    logger.info(f"Token2wav loaded in {time.time()-t0:.1f}s")

    # Load speaker data
    spk_path = os.path.join(CKPT, "spk_dict.pt")
    spk_dict = torch.load(spk_path, map_location=device)
    speaker_name = list(spk_dict.keys())[0]
    spk = spk_dict[speaker_name]
    cond = spk["cond"].float().to(device)
    ref_mel = spk["ref_mel"].float().to(device)

    logger.info(f"Using speaker: {speaker_name}")

    # Generate waveform
    t0 = time.time()
    waveform = token2wav(
        codec_filtered.to(device),
        conditioning=cond,
        reference_mel=ref_mel,
    )
    wav_time = time.time() - t0
    duration = waveform.numel() / 24000
    logger.info(f"Generated {duration:.2f}s audio in {wav_time:.2f}s")

    # Save WAV
    audio = waveform.squeeze().detach().cpu().float().numpy()
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    wav_path = "/root/test_talker_e2e.wav"
    with open(wav_path, "wb") as f:
        num_samples = len(audio_int16)
        data_size = num_samples * 2
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<I", 24000))
        f.write(struct.pack("<I", 24000 * 2))
        f.write(struct.pack("<H", 2))
        f.write(struct.pack("<H", 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(audio_int16.tobytes())

    logger.info(f"WAV saved to {wav_path} ({os.path.getsize(wav_path)} bytes)")
    logger.info("Talker-only e2e PASSED")


if __name__ == "__main__":
    test_talker_only_e2e()
    logger.info("\nAll e2e tests passed!")
