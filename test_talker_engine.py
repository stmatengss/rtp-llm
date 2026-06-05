"""Test talker loaded as LanguageCppEngine + TalkerEngineWrapper generation."""
import logging
import torch

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger("test_talker_engine")

CKPT = "/root/models/Qwen/Qwen2.5-Omni-7B"


def test_talker_engine_loading():
    """Load the talker as a full LanguageCppEngine and verify weights are accessible."""
    from rtp_llm.omni.models.qwen2_5_omni.talker import Qwen2_5OmniTalker
    from rtp_llm.config.engine_config import EngineConfig
    from rtp_llm.utils.model_weight import W
    from rtp_llm.ops import (
        ParallelismConfig, RuntimeConfig, FMHAConfig, DeviceResourceConfig,
        MoeConfig, NcclCommConfig, PDSepConfig, ConcurrencyConfig,
        ProfilingDebugLoggingConfig, HWKernelConfig, ModelSpecificConfig,
        SpeculativeExecutionConfig, CacheStoreConfig, MiscellaneousConfig,
        ArpcConfig, GrpcConfig,
    )
    from rtp_llm.config.kv_cache_config import KVCacheConfig
    from rtp_llm.config.py_config_modules import ServerConfig, LoadConfig

    logger.info("=== Testing talker engine loading ===")

    # Create talker config
    talker_config = Qwen2_5OmniTalker._create_config(CKPT)
    talker_config.ckpt_path = CKPT
    talker_config.tokenizer_path = CKPT
    talker_config.model_type = "qwen2_5_omni_talker"
    talker_config.max_seq_len = 2048
    talker_config.use_kvcache = True
    talker_config.phy2log_path = ""

    logger.info(
        f"Talker config: hidden={talker_config.hidden_size}, "
        f"layers={talker_config.num_layers}, "
        f"vocab={talker_config.vocab_size}, "
        f"embed_size={talker_config.embedding_size}, "
        f"heads={talker_config.attn_config.head_num}, "
        f"kv_heads={talker_config.attn_config.kv_head_num}, "
        f"head_dim={talker_config.attn_config.size_per_head}"
    )

    assert talker_config.hidden_size == 896
    assert talker_config.num_layers == 24
    assert talker_config.vocab_size == 8448
    assert talker_config.embedding_size == 3584
    assert talker_config.attn_config.head_num == 12
    assert talker_config.attn_config.kv_head_num == 4
    assert talker_config.attn_config.size_per_head == 128

    # Create engine config with all default sub-configs
    engine_config = EngineConfig(
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
    talker_config.init_precision_config(
        kv_cache_config=engine_config.kv_cache_config, act_type=None
    )

    # Load the model via from_config
    import inspect
    model_cls = Qwen2_5OmniTalker
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
    sig = inspect.signature(model_cls.from_config)
    if 'load_python_model' in sig.parameters:
        from_config_kwargs['load_python_model'] = True
    if 'skip_python_model' in sig.parameters:
        from_config_kwargs['skip_python_model'] = False

    logger.info("Loading talker model via from_config...")
    stage_model = model_cls.from_config(**from_config_kwargs)
    logger.info(f"Model loaded: {type(stage_model).__name__}")

    # Check weight access
    weight = stage_model.weight
    embed = weight.get_global_weight(W.embedding)
    lm_head = weight.get_global_weight(W.lm_head)
    proj_w = weight.get_global_weight("thinker_to_talker_proj.weight")
    proj_b = weight.get_global_weight("thinker_to_talker_proj.bias")

    logger.info(f"embed_tokens: {embed.shape}")
    logger.info(f"codec_head (W.lm_head): {lm_head.shape}")
    logger.info(f"proj_weight: {proj_w.shape}")
    logger.info(f"proj_bias: {proj_b.shape}")

    assert embed.shape == (8448, 3584), f"Bad embed: {embed.shape}"
    assert lm_head.shape == (8448, 896), f"Bad lm_head: {lm_head.shape}"
    assert proj_w.shape == (896, 3584), f"Bad proj_w: {proj_w.shape}"
    assert proj_b.shape == (896,), f"Bad proj_b: {proj_b.shape}"

    # Check layer weights
    layer_w = weight.weights
    logger.info(f"Layer weights available: {len(layer_w)} layers")
    assert len(layer_w) == 24, f"Expected 24 layers, got {len(layer_w)}"

    layer0 = layer_w[0]
    logger.info(f"Layer 0 keys: {list(layer0.keys())[:10]}...")

    logger.info("Talker engine loading PASSED")
    return stage_model


def test_talker_wrapper_with_engine(stage_model=None):
    """Test TalkerEngineWrapper with actual engine-loaded weights."""
    from rtp_llm.omni.models.qwen2_5_omni.talker_engine_wrapper import TalkerEngineWrapper

    if stage_model is None:
        stage_model = test_talker_engine_loading()

    logger.info("=== Testing TalkerEngineWrapper ===")

    # TalkerEngineWrapper expects engine.model to have weight, device, model_config
    class FakeEngine:
        def __init__(self, model):
            self.model = model

    wrapper = TalkerEngineWrapper(FakeEngine(stage_model))

    logger.info(
        f"Wrapper: embed={wrapper.embed_tokens_weight.shape}, "
        f"proj={wrapper.proj_weight.shape}, "
        f"codec_head={wrapper.codec_head_weight.shape}, "
        f"layers={wrapper.num_layers}, "
        f"hidden={wrapper.hidden_size}"
    )

    assert wrapper.num_layers == 24
    assert wrapper.hidden_size == 896
    assert wrapper.vocab_size == 8448

    # Test embed_and_project
    device = wrapper.proj_weight.device
    dtype = wrapper.dtype
    token_ids = torch.tensor([0, 1, 2], dtype=torch.long, device=device)
    thinker_hs = torch.randn(3, 3584, dtype=dtype, device=device)

    projected = wrapper._embed_and_project(token_ids, thinker_hs)
    assert projected.shape == (3, 896), f"Bad proj shape: {projected.shape}"
    assert not projected.isnan().any(), "NaN in projected output"
    logger.info(f"embed_and_project: {projected.shape} OK")

    # Test single layer forward
    hidden = projected
    hidden_out, kv = wrapper._layer_forward(hidden, 0, None, 0)
    assert hidden_out.shape == (3, 896), f"Bad layer output: {hidden_out.shape}"
    assert not hidden_out.isnan().any(), "NaN in layer output"
    logger.info(f"layer_forward: {hidden_out.shape} OK, kv[0]={kv[0].shape}, kv[1]={kv[1].shape}")

    logger.info("TalkerEngineWrapper PASSED")
    return wrapper


def test_talker_generation(wrapper=None):
    """Test short generation through the TalkerEngineWrapper."""
    from rtp_llm.omni.models.qwen2_5_omni.talker_engine_wrapper import TalkerEngineWrapper

    if wrapper is None:
        stage_model = test_talker_engine_loading()
        wrapper = test_talker_wrapper_with_engine(stage_model)

    logger.info("=== Testing talker generation ===")

    device = wrapper.proj_weight.device
    dtype = wrapper.dtype

    # Simulate thinker hidden states for 10 tokens
    num_thinker_tokens = 10
    thinker_hs = torch.randn(num_thinker_tokens, 3584, dtype=dtype, device=device)

    # Initial token (e.g., speaker BOS)
    initial_tokens = torch.tensor([8293], dtype=torch.long, device=device)

    # Generate a short sequence
    max_tokens = 20
    logger.info(f"Generating {max_tokens} tokens...")
    codec_tokens = wrapper.generate(
        thinker_hidden_states=thinker_hs,
        initial_token_ids=initial_tokens,
        max_new_tokens=max_tokens,
        eos_token_id=8294,
    )

    logger.info(f"Generated tokens shape: {codec_tokens.shape}")
    logger.info(f"Generated tokens: {codec_tokens[0].tolist()}")
    assert codec_tokens.shape[0] == 1
    assert codec_tokens.shape[1] <= max_tokens
    assert codec_tokens.shape[1] > 0

    logger.info("Talker generation PASSED")


if __name__ == "__main__":
    stage_model = test_talker_engine_loading()
    wrapper = test_talker_wrapper_with_engine(stage_model)
    test_talker_generation(wrapper)
    logger.info("\nAll tests passed!")
