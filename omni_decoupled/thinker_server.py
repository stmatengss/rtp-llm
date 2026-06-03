"""Decoupled omni: thinker server process.

Loads only the thinker engine (no talker, no token2wav). Reads a JSON
request from stdin describing a prompt; writes a JSON+binary response
file containing generated text, token_ids, and per-token hidden_states
(np.float16 array, base64-encoded).

Protocol:
  Request (stdin, single JSON line):
    {"prompt": "...", "max_new_tokens": 64, "audio_path": null}
  Response (--out file):
    {"text": "...", "token_ids": [int...], "hidden_states_b64": "...",
     "hidden_shape": [N, 3584], "wall_ms": float}

Usage:
  CUDA_VISIBLE_DEVICES=5 python -m omni_decoupled.thinker_server --out /tmp/thinker_out.json
"""
import argparse
import base64
import inspect
import json
import logging
import os
import sys
import time

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="thinker-server %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def make_engine_config(kv_cache_mb=4096, start_port=-100):
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
    sc.start_port = start_port
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


def make_thinker():
    from rtp_llm.omni.models.qwen2_5_omni.thinker import Qwen2_5OmniThinker
    from rtp_llm.config.py_config_modules import VitConfig
    from rtp_llm.async_decoder_engine.engine_creator import create_engine

    ec = make_engine_config()
    cfg = Qwen2_5OmniThinker._create_config(os.environ.get("OMNI_CKPT", "/root/models/Qwen/Qwen2.5-Omni-7B"))
    cfg.ckpt_path = cfg.ckpt_path
    cfg.tokenizer_path = cfg.ckpt_path
    cfg.model_type = "qwen2_5_omni_thinker"
    cfg.max_seq_len = 4096
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
        vit_config=VitConfig(),
        merge_lora=False,
        device_resource_config=ec.device_resource_config,
        force_cpu_load_weights=ec.load_config.force_cpu_load_weights,
    )
    sig = inspect.signature(Qwen2_5OmniThinker.from_config)
    if 'load_python_model' in sig.parameters:
        kwargs['load_python_model'] = True
    if 'skip_python_model' in sig.parameters:
        kwargs['skip_python_model'] = False

    model = Qwen2_5OmniThinker.from_config(**kwargs)
    engine = create_engine(
        model=model, engine_config=ec,
        alog_conf_path=ec.profiling_debug_logging_config.ft_alog_conf_path,
        world_info=None,
    )
    engine.start()
    return engine, model


def main():
    # CLI args can't be used: rtp_llm imports trigger argparse on sys.argv.
    # Inputs/outputs via env vars instead.
    out_path = os.environ.get("OMNI_OUT")
    if not out_path:
        print("ERROR: set OMNI_OUT env var to output JSON path", file=sys.stderr)
        return 2
    req_inline = os.environ.get("OMNI_REQUEST")
    if req_inline:
        req = json.loads(req_inline)
    else:
        req = json.loads(sys.stdin.read())

    prompt = req["prompt"]
    max_new = int(req.get("max_new_tokens", 64))

    logger.info("loading thinker...")
    t_load = time.perf_counter()
    engine, model = make_thinker()
    logger.info(f"thinker loaded in {time.perf_counter() - t_load:.1f}s")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.environ.get("OMNI_CKPT", "/root/models/Qwen/Qwen2.5-Omni-7B"))
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt_text)
    input_ids = torch.tensor(prompt_ids, dtype=torch.int32)
    eos = tokenizer.eos_token_id or 151643

    t0 = time.perf_counter()
    token_ids, hidden_states = engine.rtp_llm_op_.generate(
        input_ids, max_new_tokens=max_new, eos_token_id=eos, return_hidden_states=True,
    )
    wall_ms = (time.perf_counter() - t0) * 1000

    out_ids = token_ids[0].cpu().tolist() if token_ids.numel() else []
    text = tokenizer.decode(out_ids, skip_special_tokens=True)

    # Hidden states as fp16 little-endian bytes, base64-encoded
    hs_fp16 = hidden_states.to(torch.float16).cpu().contiguous().numpy()
    hs_bytes = hs_fp16.tobytes()
    hs_b64 = base64.b64encode(hs_bytes).decode("ascii")

    payload = {
        "text": text,
        "token_ids": out_ids,
        "hidden_states_b64": hs_b64,
        "hidden_shape": list(hs_fp16.shape),
        "hidden_dtype": "float16",
        "wall_ms": wall_ms,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)
    logger.info(f"wrote {out_path}: text={text!r} hs.shape={hs_fp16.shape} wall={wall_ms:.0f}ms")
    # Skip engine shutdown — known bad_alloc race; just exit
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
