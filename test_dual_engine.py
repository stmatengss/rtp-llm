"""Test loading both thinker and talker as LanguageCppEngine instances."""
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger("test_dual_engine")

CKPT = "/root/models/Qwen/Qwen2.5-Omni-7B"


def test_stage_config_creation():
    """Test that _create_config produces correct configs for both stages."""
    from rtp_llm.omni.models.qwen2_5_omni.thinker import Qwen2_5OmniThinker
    from rtp_llm.omni.models.qwen2_5_omni.talker import Qwen2_5OmniTalker

    logger.info("=== Testing thinker _create_config ===")
    thinker_config = Qwen2_5OmniThinker._create_config(CKPT)
    logger.info(f"Thinker: hidden={thinker_config.hidden_size}, layers={thinker_config.num_layers}, "
                f"vocab={thinker_config.vocab_size}, heads={thinker_config.attn_config.head_num}, "
                f"kv_heads={thinker_config.attn_config.kv_head_num}, "
                f"head_dim={thinker_config.attn_config.size_per_head}")

    assert thinker_config.hidden_size == 3584
    assert thinker_config.num_layers == 28
    assert thinker_config.vocab_size == 152064
    logger.info("Thinker config PASSED")

    logger.info("=== Testing talker _create_config ===")
    talker_config = Qwen2_5OmniTalker._create_config(CKPT)
    logger.info(f"Talker: hidden={talker_config.hidden_size}, layers={talker_config.num_layers}, "
                f"vocab={talker_config.vocab_size}, heads={talker_config.attn_config.head_num}, "
                f"kv_heads={talker_config.attn_config.kv_head_num}, "
                f"head_dim={talker_config.attn_config.size_per_head}, "
                f"embed_size={talker_config.embedding_size}")

    assert talker_config.hidden_size == 896
    assert talker_config.num_layers == 24
    assert talker_config.vocab_size == 8448
    assert talker_config.attn_config.head_num == 12
    assert talker_config.attn_config.kv_head_num == 4
    assert talker_config.attn_config.size_per_head == 128
    assert talker_config.embedding_size == 3584
    logger.info("Talker config PASSED")

    logger.info("=== Verifying configs are different ===")
    assert thinker_config.hidden_size != talker_config.hidden_size
    assert thinker_config.num_layers != talker_config.num_layers
    assert thinker_config.vocab_size != talker_config.vocab_size
    logger.info("Config difference PASSED")


def test_weight_class_registration():
    """Test that both model types are registered and weight classes exist."""
    from rtp_llm.model_factory import ModelFactory

    logger.info("=== Testing model registration ===")

    thinker_cls = ModelFactory.get_model_cls("qwen2_5_omni_thinker")
    talker_cls = ModelFactory.get_model_cls("qwen2_5_omni_talker")

    logger.info(f"Thinker cls: {thinker_cls.__name__}")
    logger.info(f"Talker cls: {talker_cls.__name__}")

    thinker_weight_cls = thinker_cls.get_weight_cls()
    talker_weight_cls = talker_cls.get_weight_cls()

    logger.info(f"Thinker weight cls: {thinker_weight_cls.__name__}")
    logger.info(f"Talker weight cls: {talker_weight_cls.__name__}")

    assert thinker_cls.__name__ == "Qwen2_5OmniThinker"
    assert talker_cls.__name__ == "Qwen2_5OmniTalker"
    assert thinker_weight_cls.__name__ == "Qwen2_5OmniThinkerWeight"
    assert talker_weight_cls.__name__ == "Qwen2_5OmniTalkerWeight"
    logger.info("Registration PASSED")


def test_omni_engine_construction():
    """Test OmniEngine construction with pipeline config."""
    from rtp_llm.omni.models.qwen2_5_omni.pipeline import QWEN2_5_OMNI_PIPELINE
    from rtp_llm.omni.engine.omni_engine import OmniEngine

    logger.info("=== Testing OmniEngine construction ===")
    engine = OmniEngine.from_pipeline_config(QWEN2_5_OMNI_PIPELINE)
    assert engine.num_stages == 3
    assert engine._primary_engine is None
    assert engine._thinker_engine is None
    assert engine._talker_engine is None
    logger.info(f"OmniEngine created with {engine.num_stages} stages")
    logger.info("Construction PASSED")


def test_stage_model_config_creation():
    """Test _create_stage_model_config produces correct talker config."""
    from rtp_llm.omni.models.qwen2_5_omni.pipeline import QWEN2_5_OMNI_PIPELINE
    from rtp_llm.omni.models.qwen2_5_omni.thinker import Qwen2_5OmniThinker
    from rtp_llm.omni.engine.omni_engine import OmniEngine
    from rtp_llm.ops import TaskType

    logger.info("=== Testing _create_stage_model_config ===")

    engine = OmniEngine.from_pipeline_config(QWEN2_5_OMNI_PIPELINE)

    # Create main (thinker) config as if it came from ModelFactory
    main_config = Qwen2_5OmniThinker._create_config(CKPT)
    main_config.model_type = "qwen2_5_omni_thinker"
    main_config.max_seq_len = 8192
    main_config.task_type = TaskType.LANGUAGE_MODEL
    main_config.use_kvcache = True

    # Get the talker stage
    talker_stage = QWEN2_5_OMNI_PIPELINE.get_stage(1)
    assert talker_stage.model_stage == "talker"

    talker_config = engine._create_stage_model_config(
        talker_stage, main_config, None
    )

    logger.info(f"Talker stage config: hidden={talker_config.hidden_size}, "
                f"layers={talker_config.num_layers}, vocab={talker_config.vocab_size}, "
                f"embed_size={talker_config.embedding_size}, "
                f"model_type={talker_config.model_type}")

    assert talker_config.hidden_size == 896
    assert talker_config.num_layers == 24
    assert talker_config.vocab_size == 8448
    assert talker_config.embedding_size == 3584
    assert talker_config.model_type == "qwen2_5_omni_talker"
    assert talker_config.max_seq_len == 8192
    assert talker_config.ckpt_path == CKPT
    logger.info("Stage model config PASSED")


if __name__ == "__main__":
    test_stage_config_creation()
    test_weight_class_registration()
    test_omni_engine_construction()
    test_stage_model_config_creation()
    logger.info("\nAll tests passed!")
