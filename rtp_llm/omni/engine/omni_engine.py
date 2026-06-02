import logging
from typing import Any, Dict, Optional

import torch

from rtp_llm.omni.config.stage_config import (
    OmniPipelineConfig,
    OmniStageConfig,
    StageExecutionType,
)
from rtp_llm.omni.engine.orchestrator import OmniOrchestrator
from rtp_llm.omni.engine.output_processor import OmniOutputProcessor
from rtp_llm.omni.engine.stage_connector import SharedMemoryConnector, StageConnector, StageOutput
from rtp_llm.omni.engine.stage_pool import OmniStagePool

logger = logging.getLogger(__name__)


class Token2WavStub:
    def __init__(self, stage_config: OmniStageConfig):
        self.stage_config = stage_config

    def forward(self, codec_tokens):
        logger.warning("Token2Wav stub: returning empty audio (Phase 4)")
        return StageOutput(
            audio_waveform=torch.zeros(1, 16000),
            metadata={"stub": True},
        )


class OmniEngine:
    """Multi-stage engine for omni models (e.g. thinker → talker → token2wav).

    Implements the same interface as BaseEngine (duck-typed) so it can be
    used by BackendManager without inheritance from BaseEngine (which
    requires a BaseModel in __init__).
    """

    def __init__(
        self,
        pipeline_config: OmniPipelineConfig,
        connector: Optional[StageConnector] = None,
        model_config: Any = None,
        engine_config: Any = None,
    ):
        self.pipeline_config = pipeline_config
        self.config = model_config
        self.model_config = model_config
        self.engine_config = engine_config
        self.connector = connector or SharedMemoryConnector()
        self.output_processor = OmniOutputProcessor()

        self.stage_pools: Dict[int, OmniStagePool] = {}
        for stage_config in pipeline_config.stages:
            self.stage_pools[stage_config.stage_id] = OmniStagePool(
                stage_config=stage_config
            )

        self.orchestrator = OmniOrchestrator(
            pipeline_config=pipeline_config,
            connector=self.connector,
            stage_pools=self.stage_pools,
        )

        self.stage_engines: Dict[int, Any] = {}
        self._primary_engine = None
        self.started = False

        logger.info(
            f"OmniEngine created for {pipeline_config.model_type} "
            f"with {len(pipeline_config.stages)} stages"
        )

    @property
    def num_stages(self) -> int:
        return len(self.pipeline_config.stages)

    @property
    def task_type(self):
        if self._primary_engine is not None:
            return self._primary_engine.config.task_type
        from rtp_llm.ops import TaskType
        return TaskType.LANGUAGE_MODEL

    @property
    def default_generate_config(self):
        if self._primary_engine is not None:
            return self._primary_engine.default_generate_config
        return None

    @property
    def role_type(self) -> str:
        if self._primary_engine is not None and hasattr(self._primary_engine, 'role_type'):
            return self._primary_engine.role_type
        return "omni"

    def get_final_output_types(self) -> Dict[str, int]:
        result = {}
        for stage in self.pipeline_config.stages:
            if stage.final_output and stage.final_output_type:
                result[stage.final_output_type] = stage.stage_id
        return result

    def initialize_stages(
        self,
        model_config: Any,
        engine_config: Any,
        world_info: Any,
        vit_config: Any = None,
        merge_lora: bool = False,
    ) -> None:
        """Create per-stage sub-engines.

        For LLM_AR stages: creates a full LanguageCppEngine via the standard path.
        For LLM_GENERATION stages (token2wav): creates a stub.

        Currently only the first LLM_AR stage (thinker) is loaded. Other stages
        are deferred due to GPU memory constraints (Phase 4).
        """
        self.model_config = model_config
        self.config = model_config
        self.engine_config = engine_config

        from rtp_llm.model_factory import ModelFactory

        primary_created = False
        for stage in self.pipeline_config.stages:
            if stage.execution_type == StageExecutionType.LLM_AR:
                if primary_created:
                    logger.info(
                        f"Stage {stage.stage_id} ({stage.model_stage}) "
                        f"deferred — only primary stage loaded in Phase 3"
                    )
                    continue

                logger.info(
                    f"Initializing LLM_AR stage {stage.stage_id} "
                    f"({stage.model_stage}) with model_type={stage.model_type}"
                )

                stage_model_type = stage.model_type
                if stage_model_type is None:
                    raise ValueError(
                        f"Stage {stage.stage_id} ({stage.model_stage}) "
                        f"has no model_type configured"
                    )

                model_cls = ModelFactory.get_model_cls(stage_model_type)

                from_config_kwargs = dict(
                    model_config=model_config,
                    parallelism_config=engine_config.parallelism_config,
                    hw_kernel_config=engine_config.hw_kernel_config,
                    kv_cache_config=engine_config.kv_cache_config,
                    fmha_config=engine_config.fmha_config,
                    moe_config=engine_config.moe_config,
                    load_method=engine_config.load_config.load_method,
                    max_generate_batch_size=engine_config.runtime_config.max_generate_batch_size,
                    vit_config=vit_config,
                    merge_lora=merge_lora,
                    device_resource_config=engine_config.device_resource_config,
                    force_cpu_load_weights=engine_config.load_config.force_cpu_load_weights,
                )
                import inspect
                sig = inspect.signature(model_cls.from_config)
                if 'load_python_model' in sig.parameters:
                    from_config_kwargs['load_python_model'] = True
                if 'skip_python_model' in sig.parameters:
                    from_config_kwargs['skip_python_model'] = False
                stage_model = model_cls.from_config(**from_config_kwargs)

                alog_conf_path = engine_config.profiling_debug_logging_config.ft_alog_conf_path
                from rtp_llm.async_decoder_engine.engine_creator import create_engine
                sub_engine = create_engine(
                    model=stage_model,
                    engine_config=engine_config,
                    alog_conf_path=alog_conf_path,
                    world_info=world_info,
                )
                self.stage_engines[stage.stage_id] = sub_engine
                if not primary_created:
                    self._primary_engine = sub_engine
                    primary_created = True

                logger.info(
                    f"Stage {stage.stage_id} ({stage.model_stage}) engine created"
                )

            elif stage.execution_type == StageExecutionType.LLM_GENERATION:
                logger.info(
                    f"Stage {stage.stage_id} ({stage.model_stage}) "
                    f"using Token2WavStub (Phase 4 will implement)"
                )
                self.stage_engines[stage.stage_id] = Token2WavStub(stage)

            else:
                logger.warning(
                    f"Stage {stage.stage_id} ({stage.model_stage}) "
                    f"has unsupported execution type: {stage.execution_type}"
                )

    def start(self) -> None:
        for stage_id, engine in self.stage_engines.items():
            if hasattr(engine, 'start'):
                stage = self.pipeline_config.get_stage(stage_id)
                logger.info(f"Starting stage {stage_id} ({stage.model_stage})")
                engine.start()
        self.started = True
        logger.info(
            f"OmniEngine started with {len(self.stage_engines)} active stages"
        )

    def stop(self) -> None:
        self.started = False
        for stage_id, engine in self.stage_engines.items():
            if hasattr(engine, 'stop'):
                stage = self.pipeline_config.get_stage(stage_id)
                logger.info(f"Stopping stage {stage_id} ({stage.model_stage})")
                engine.stop()
        logger.info("OmniEngine stopped")

    def ready(self) -> bool:
        if not self.started:
            return False
        for stage_id, engine in self.stage_engines.items():
            if hasattr(engine, 'ready') and not engine.ready():
                return False
        return True

    @classmethod
    def from_pipeline_config(
        cls,
        pipeline_config: OmniPipelineConfig,
        model_config: Any = None,
        engine_config: Any = None,
    ) -> "OmniEngine":
        return cls(
            pipeline_config=pipeline_config,
            model_config=model_config,
            engine_config=engine_config,
        )
