"""Decoupled omni: talker client process.

Reads the JSON+b64 payload written by thinker_server.py, loads ONLY the
talker engine + token2wav (no thinker weights), generates codec tokens
conditioned on the received hidden_states, and writes a WAV.

Usage:
  CUDA_VISIBLE_DEVICES=6 python -m omni_decoupled.talker_client \
      --thinker-out /tmp/thinker_out.json --wav-out /tmp/decoupled.wav
"""
import argparse
import base64
import inspect
import json
import logging
import os
import struct
import sys
import time

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="talker-client %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TALKER_CODEC_BOS = 8293
TALKER_CODEC_EOS = 8294


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


def make_talker():
    from rtp_llm.omni.models.qwen2_5_omni.talker import Qwen2_5OmniTalker
    from rtp_llm.async_decoder_engine.engine_creator import create_engine
    ec = make_engine_config()
    cfg = Qwen2_5OmniTalker._create_config(os.environ.get("OMNI_CKPT", "/root/models/Qwen/Qwen2.5-Omni-7B"))
    cfg.ckpt_path = cfg.ckpt_path
    cfg.tokenizer_path = cfg.ckpt_path
    cfg.model_type = "qwen2_5_omni_talker"
    cfg.max_seq_len = 2048
    cfg.use_kvcache = True
    cfg.phy2log_path = ""
    cfg.init_precision_config(kv_cache_config=ec.kv_cache_config, act_type=None)

    kwargs = dict(
        model_config=cfg,
        parallelism_config=ec.parallelism_config,
        hw_kernel_config=ec.hw_kernel_config,
        kv_cache_config=ec.kv_cache_config,
        fmha_config=ec.fmha_config,
        moe_config=ec.moe_config,
        load_method=ec.load_config.load_method,
        max_generate_batch_size=ec.runtime_config.max_generate_batch_size,
        vit_config=None,
        merge_lora=False,
        device_resource_config=ec.device_resource_config,
        force_cpu_load_weights=ec.load_config.force_cpu_load_weights,
    )
    sig = inspect.signature(Qwen2_5OmniTalker.from_config)
    if 'load_python_model' in sig.parameters:
        kwargs['load_python_model'] = True
    if 'skip_python_model' in sig.parameters:
        kwargs['skip_python_model'] = False

    model = Qwen2_5OmniTalker.from_config(**kwargs)
    engine = create_engine(
        model=model, engine_config=ec,
        alog_conf_path=ec.profiling_debug_logging_config.ft_alog_conf_path,
        world_info=None,
    )
    engine.start()
    return engine, model


def save_wav(waveform, path, sample_rate=24000):
    a = waveform.squeeze().detach().cpu().float().numpy()
    a = np.clip(a, -1.0, 1.0)
    s = (a * 32767).astype(np.int16)
    with open(path, "wb") as f:
        n = len(s); ds = n * 2
        f.write(b"RIFF"); f.write(struct.pack("<I", 36 + ds)); f.write(b"WAVE")
        f.write(b"fmt "); f.write(struct.pack("<I", 16)); f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", 1)); f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * 2)); f.write(struct.pack("<H", 2))
        f.write(struct.pack("<H", 16)); f.write(b"data"); f.write(struct.pack("<I", ds))
        f.write(s.tobytes())
    return n / sample_rate


def main():
    # CLI args can't be used: rtp_llm imports trigger argparse on sys.argv.
    thinker_out = os.environ.get("OMNI_THINKER_OUT")
    wav_out = os.environ.get("OMNI_WAV_OUT")
    if not thinker_out or not wav_out:
        print("ERROR: set OMNI_THINKER_OUT and OMNI_WAV_OUT env vars", file=sys.stderr)
        return 2

    logger.info(f"reading thinker payload {thinker_out}")
    with open(thinker_out) as f:
        payload = json.load(f)
    text = payload["text"]
    hs_shape = payload["hidden_shape"]
    hs_bytes = base64.b64decode(payload["hidden_states_b64"])
    hs_np = np.frombuffer(hs_bytes, dtype=np.float16).reshape(hs_shape)
    hs = torch.from_numpy(hs_np.copy())
    logger.info(f"received hidden_states: shape={hs.shape} text={text!r}")

    logger.info("loading talker...")
    t_load = time.perf_counter()
    talker_engine, talker_model = make_talker()
    logger.info(f"talker loaded in {time.perf_counter() - t_load:.1f}s")

    device = "cuda:0"
    talker_model.py_model.set_thinker_hidden_states(hs.to(device=device, dtype=torch.bfloat16))
    initial = torch.tensor([TALKER_CODEC_BOS], dtype=torch.int32)
    t_gen = time.perf_counter()
    codec = talker_engine.rtp_llm_op_.generate(initial, max_new_tokens=200, eos_token_id=TALKER_CODEC_EOS)
    talker_ms = (time.perf_counter() - t_gen) * 1000
    talker_model.py_model.clear_thinker_hidden_states()
    logger.info(f"talker generated {codec.shape[1]} codec tokens in {talker_ms:.0f}ms")

    mask = codec[0] < 8292
    codec_filtered = codec[0][mask].unsqueeze(0).cpu()

    # token2wav on same device
    from rtp_llm.omni.models.qwen2_5_omni.token2wav_model import Token2WavModel
    audio_device = device
    talker_engine.stop()
    import gc
    del talker_engine, talker_model
    gc.collect(); torch.cuda.empty_cache()

    token2wav = Token2WavModel.from_pretrained(
        os.environ.get("OMNI_CKPT", "/root/models/Qwen/Qwen2.5-Omni-7B"), device=audio_device)
    spk_dict = torch.load(
        os.path.join(os.environ.get("OMNI_CKPT", "/root/models/Qwen/Qwen2.5-Omni-7B"), "spk_dict.pt"),
        map_location=audio_device)
    spk = next(iter(spk_dict.values()))
    cond = spk["cond"].float().to(audio_device)
    ref_mel = spk["ref_mel"].float().to(audio_device)
    waveform = token2wav(codec_filtered.to(audio_device), conditioning=cond, reference_mel=ref_mel)
    dur = save_wav(waveform, wav_out)
    logger.info(f"saved {wav_out}: {dur:.2f}s")
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
