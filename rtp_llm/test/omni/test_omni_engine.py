import unittest

from rtp_llm.omni.config.stage_config import (
    OmniPipelineConfig,
    OmniStageConfig,
    StageExecutionType,
)
from rtp_llm.omni.engine.omni_engine import OmniEngine


class TestOmniEngine(unittest.TestCase):
    def _make_pipeline_config(self):
        return OmniPipelineConfig(
            model_type="test_omni",
            model_arch="TestOmniArch",
            stages=(
                OmniStageConfig(
                    stage_id=0,
                    model_stage="thinker",
                    execution_type=StageExecutionType.LLM_AR,
                    model_cls="TestThinker",
                    input_sources=(),
                    final_output=True,
                    final_output_type="text",
                ),
                OmniStageConfig(
                    stage_id=1,
                    model_stage="talker",
                    execution_type=StageExecutionType.LLM_AR,
                    model_cls="TestTalker",
                    input_sources=(0,),
                    final_output=True,
                    final_output_type="audio",
                ),
            ),
        )

    def test_create_omni_engine(self):
        config = self._make_pipeline_config()
        engine = OmniEngine(pipeline_config=config)
        self.assertEqual(engine.pipeline_config.model_type, "test_omni")
        self.assertEqual(engine.num_stages, 2)

    def test_get_final_output_types(self):
        config = self._make_pipeline_config()
        engine = OmniEngine(pipeline_config=config)
        output_types = engine.get_final_output_types()
        self.assertEqual(output_types, {"text": 0, "audio": 1})

    def test_stage_pools_initialized(self):
        config = self._make_pipeline_config()
        engine = OmniEngine(pipeline_config=config)
        self.assertEqual(len(engine.stage_pools), 2)
        self.assertIn(0, engine.stage_pools)
        self.assertIn(1, engine.stage_pools)

    def test_orchestrator_initialized(self):
        config = self._make_pipeline_config()
        engine = OmniEngine(pipeline_config=config)
        self.assertIsNotNone(engine.orchestrator)

    def test_connector_initialized(self):
        config = self._make_pipeline_config()
        engine = OmniEngine(pipeline_config=config)
        self.assertIsNotNone(engine.connector)

    def test_register_and_get_stage_engine(self):
        config = self._make_pipeline_config()
        engine = OmniEngine(pipeline_config=config)
        mock_engine = object()
        engine.register_stage_engine(0, mock_engine)
        self.assertIs(engine.get_stage_engine(0), mock_engine)
        self.assertIsNone(engine.get_stage_engine(1))

    def test_get_execution_order(self):
        config = self._make_pipeline_config()
        engine = OmniEngine(pipeline_config=config)
        self.assertEqual(engine.get_execution_order(), [0, 1])

    def test_transform_stage_output_no_processor(self):
        from rtp_llm.omni.engine.stage_connector import StageOutput

        config = self._make_pipeline_config()
        engine = OmniEngine(pipeline_config=config)
        output = StageOutput(token_ids=[1, 2, 3])
        result = engine.transform_stage_output(0, output)
        self.assertIs(result, output)

    def test_transform_stage_output_with_processor(self):
        from rtp_llm.omni.engine.stage_connector import StageOutput

        config = OmniPipelineConfig(
            model_type="test_proc",
            model_arch="TestProc",
            stages=(
                OmniStageConfig(
                    stage_id=0,
                    model_stage="entry",
                    execution_type=StageExecutionType.LLM_AR,
                    model_cls="TestEntry",
                ),
                OmniStageConfig(
                    stage_id=1,
                    model_stage="next",
                    execution_type=StageExecutionType.LLM_AR,
                    model_cls="TestNext",
                    input_sources=(0,),
                    custom_process_input_func="rtp_llm.omni.models.qwen2_5_omni.stage_processors.thinker2talker",
                ),
            ),
        )
        engine = OmniEngine(pipeline_config=config)
        output = StageOutput(
            token_ids=[1, 2], metadata={"text": "hello"}
        )
        result = engine.transform_stage_output(1, output)
        self.assertEqual(result.metadata["source_text"], "hello")


if __name__ == "__main__":
    unittest.main()
