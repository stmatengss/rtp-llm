import unittest

from rtp_llm.omni.config.pipeline_registry import OmniPipelineRegistry
from rtp_llm.omni.config.stage_config import StageExecutionType


class TestQwen25OmniPipeline(unittest.TestCase):
    def test_pipeline_registered(self):
        from rtp_llm.omni.models.qwen2_5_omni import pipeline  # noqa: F401

        config = OmniPipelineRegistry.get("qwen2_5_omni")
        self.assertIsNotNone(config)

    def test_pipeline_has_three_stages(self):
        from rtp_llm.omni.models.qwen2_5_omni import pipeline  # noqa: F401

        config = OmniPipelineRegistry.get("qwen2_5_omni")
        self.assertEqual(len(config.stages), 3)

    def test_pipeline_model_arch(self):
        from rtp_llm.omni.models.qwen2_5_omni import pipeline  # noqa: F401

        config = OmniPipelineRegistry.get("qwen2_5_omni")
        self.assertEqual(config.model_arch, "Qwen2_5OmniModel")

    def test_thinker_stage(self):
        from rtp_llm.omni.models.qwen2_5_omni import pipeline  # noqa: F401

        config = OmniPipelineRegistry.get("qwen2_5_omni")
        thinker = config.get_stage(0)
        self.assertEqual(thinker.model_stage, "thinker")
        self.assertEqual(thinker.execution_type, StageExecutionType.LLM_AR)
        self.assertEqual(thinker.model_cls, "Qwen2_5OmniThinker")
        self.assertTrue(thinker.final_output)
        self.assertEqual(thinker.final_output_type, "text")
        self.assertTrue(thinker.requires_multimodal_data)
        self.assertEqual(thinker.engine_output_type, "latent")
        self.assertEqual(thinker.input_sources, ())

    def test_talker_stage(self):
        from rtp_llm.omni.models.qwen2_5_omni import pipeline  # noqa: F401

        config = OmniPipelineRegistry.get("qwen2_5_omni")
        talker = config.get_stage(1)
        self.assertEqual(talker.model_stage, "talker")
        self.assertEqual(talker.execution_type, StageExecutionType.LLM_AR)
        self.assertEqual(talker.model_cls, "Qwen2_5OmniTalker")
        self.assertEqual(talker.input_sources, (0,))
        self.assertEqual(talker.stage_processor, "qwen2_5_omni.thinker2talker")

    def test_token2wav_stage(self):
        from rtp_llm.omni.models.qwen2_5_omni import pipeline  # noqa: F401

        config = OmniPipelineRegistry.get("qwen2_5_omni")
        t2w = config.get_stage(2)
        self.assertEqual(t2w.model_stage, "token2wav")
        self.assertEqual(t2w.execution_type, StageExecutionType.LLM_GENERATION)
        self.assertEqual(t2w.model_cls, "Qwen2_5OmniToken2Wav")
        self.assertEqual(t2w.input_sources, (1,))
        self.assertTrue(t2w.final_output)
        self.assertEqual(t2w.final_output_type, "audio")
        self.assertEqual(t2w.stage_processor, "qwen2_5_omni.talker2token2wav")

    def test_final_output_stages(self):
        from rtp_llm.omni.models.qwen2_5_omni import pipeline  # noqa: F401

        config = OmniPipelineRegistry.get("qwen2_5_omni")
        final_stages = config.get_final_output_stages()
        self.assertEqual(len(final_stages), 2)
        types = {s.final_output_type for s in final_stages}
        self.assertEqual(types, {"text", "audio"})


if __name__ == "__main__":
    unittest.main()
