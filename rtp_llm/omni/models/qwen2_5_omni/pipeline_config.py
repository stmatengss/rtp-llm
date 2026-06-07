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
            model_cls="qwen2_5_omni_thinker",
            input_sources=(),
            final_output=True,
            final_output_type="text",
            requires_multimodal_data=True,
            engine_output_type="latent",
            owns_tokenizer=True,
        ),
        OmniStageConfig(
            stage_id=1,
            model_stage="talker",
            execution_type=StageExecutionType.LLM_AR,
            model_cls="qwen2_5_omni_talker",
            input_sources=(0,),
            engine_output_type="latent",
            custom_process_input_func="rtp_llm.omni.models.qwen2_5_omni.stage_processors.thinker2talker",
            sampling_constraints={"stop_token_ids": [8294]},
        ),
        OmniStageConfig(
            stage_id=2,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            model_cls="qwen2_5_omni_token2wav",
            input_sources=(1,),
            final_output=True,
            final_output_type="audio",
            custom_process_input_func="rtp_llm.omni.models.qwen2_5_omni.stage_processors.talker2code2wav",
        ),
    ),
)
