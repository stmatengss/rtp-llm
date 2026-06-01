from rtp_llm.omni.config.pipeline_registry import OmniPipelineRegistry
from rtp_llm.omni.config.stage_config import (
    OmniPipelineConfig,
    OmniStageConfig,
    StageExecutionType,
)

QWEN2_5_OMNI_PIPELINE = OmniPipelineConfig(
    model_type="qwen2_5_omni",
    model_arch="Qwen2_5OmniModel",
    stages=(
        OmniStageConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            model_cls="Qwen2_5OmniThinker",
            input_sources=(),
            final_output=True,
            final_output_type="text",
            requires_multimodal_data=True,
            engine_output_type="latent",
        ),
        OmniStageConfig(
            stage_id=1,
            model_stage="talker",
            execution_type=StageExecutionType.LLM_AR,
            model_cls="Qwen2_5OmniTalker",
            input_sources=(0,),
            engine_output_type="latent",
            stage_processor="qwen2_5_omni.thinker2talker",
        ),
        OmniStageConfig(
            stage_id=2,
            model_stage="token2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            model_cls="Qwen2_5OmniToken2Wav",
            input_sources=(1,),
            final_output=True,
            final_output_type="audio",
            stage_processor="qwen2_5_omni.talker2token2wav",
        ),
    ),
)

OmniPipelineRegistry.register(QWEN2_5_OMNI_PIPELINE)
