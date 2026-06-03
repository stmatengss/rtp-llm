"""Verify RtpLLMOp.generate() can return real hidden states."""
import os
import sys
import inspect
import logging

import torch

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger("test_real_hidden_states")

CKPT = "/root/models/Qwen/Qwen2.5-Omni-7B"


def make_engine_config():
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
    kv.kv_cache_mem_mb = 4096
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


def main():
    from rtp_llm.omni.models.qwen2_5_omni.thinker import Qwen2_5OmniThinker
    from rtp_llm.config.py_config_modules import VitConfig
    from rtp_llm.async_decoder_engine.engine_creator import create_engine
    from transformers import AutoTokenizer

    engine_config = make_engine_config()
    config = Qwen2_5OmniThinker._create_config(CKPT)
    config.ckpt_path = CKPT
    config.tokenizer_path = CKPT
    config.model_type = "qwen2_5_omni_thinker"
    config.max_seq_len = 4096
    config.use_kvcache = True
    config.phy2log_path = ""
    config.init_precision_config(kv_cache_config=engine_config.kv_cache_config, act_type=None)

    kwargs = dict(
        model_config=config,
        parallelism_config=engine_config.parallelism_config,
        hw_kernel_config=engine_config.hw_kernel_config,
        kv_cache_config=engine_config.kv_cache_config,
        fmha_config=engine_config.fmha_config,
        moe_config=engine_config.moe_config,
        load_method=engine_config.load_config.load_method,
        max_generate_batch_size=engine_config.runtime_config.max_generate_batch_size,
        vit_config=VitConfig(),
        merge_lora=False,
        device_resource_config=engine_config.device_resource_config,
        force_cpu_load_weights=engine_config.load_config.force_cpu_load_weights,
    )
    sig = inspect.signature(Qwen2_5OmniThinker.from_config)
    if 'load_python_model' in sig.parameters:
        kwargs['load_python_model'] = True
    if 'skip_python_model' in sig.parameters:
        kwargs['skip_python_model'] = False

    model = Qwen2_5OmniThinker.from_config(**kwargs)
    engine = create_engine(
        model=model, engine_config=engine_config,
        alog_conf_path=engine_config.profiling_debug_logging_config.ft_alog_conf_path,
        world_info=None,
    )
    engine.start()

    tok = AutoTokenizer.from_pretrained(CKPT)
    prompt_ids = tok.encode("Tell me a joke.", return_tensors="pt").to(torch.int32)[0]
    eos = tok.eos_token_id or 151643

    # Call with return_hidden_states=True; result must be a 2-tuple
    result = engine.rtp_llm_op_.generate(
        prompt_ids, max_new_tokens=16, eos_token_id=eos, return_hidden_states=True
    )
    assert isinstance(result, tuple), f"expected tuple, got {type(result)}"
    assert len(result) == 2, f"expected 2-tuple, got {len(result)}"
    token_ids, hidden_states = result

    assert isinstance(token_ids, torch.Tensor), f"token_ids type: {type(token_ids)}"
    assert token_ids.dim() == 2 and token_ids.shape[0] == 1, f"token_ids shape: {token_ids.shape}"
    num_gen = token_ids.shape[1]
    assert num_gen > 0, "no tokens generated"

    assert isinstance(hidden_states, torch.Tensor), f"hidden_states type: {type(hidden_states)}"
    assert hidden_states.dim() == 2, f"hidden_states must be 2D, got shape {hidden_states.shape}"
    assert hidden_states.shape[0] == num_gen, (
        f"hidden_states rows ({hidden_states.shape[0]}) != num_gen ({num_gen})"
    )
    # thinker hidden_size is 3584
    assert hidden_states.shape[1] == 3584, (
        f"hidden_states cols ({hidden_states.shape[1]}) != 3584"
    )

    # Sanity: hidden states are NOT all zeros (would indicate they weren't populated)
    assert hidden_states.abs().sum().item() > 0, "hidden_states are all-zero"

    logger.info("PASS: tokens=%d hidden_states.shape=%s", num_gen, tuple(hidden_states.shape))
    return 0


if __name__ == "__main__":
    sys.exit(main())
