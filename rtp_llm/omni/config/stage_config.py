from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class StageExecutionType(Enum):
    LLM_AR = "llm_ar"
    LLM_GENERATION = "llm_generation"
    DIFFUSION = "diffusion"


@dataclass(frozen=True)
class OmniStageConfig:
    stage_id: int
    model_stage: str
    execution_type: StageExecutionType
    model_cls: str
    input_sources: Tuple[int, ...] = ()
    final_output: bool = False
    final_output_type: Optional[str] = None
    requires_multimodal_data: bool = False
    engine_output_type: Optional[str] = None
    custom_process_input_func: Optional[str] = None
    owns_tokenizer: bool = False
    model_subdir: Optional[str] = None
    sampling_constraints: Dict[str, Any] = field(default_factory=dict)
    engine_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OmniPipelineConfig:
    model_type: str
    model_arch: str
    stages: Tuple[OmniStageConfig, ...]

    def validate(self) -> None:
        stage_ids = {s.stage_id for s in self.stages}
        if len(stage_ids) != len(self.stages):
            raise ValueError(f"Duplicate stage_ids in pipeline {self.model_type}")
        for s in self.stages:
            if s.stage_id in s.input_sources:
                raise ValueError(
                    f"Stage {s.stage_id} references itself in input_sources"
                )
            for src in s.input_sources:
                if src not in stage_ids:
                    raise ValueError(
                        f"Stage {s.stage_id} references nonexistent "
                        f"input_source {src}"
                    )
        entry_points = [s for s in self.stages if not s.input_sources]
        if not entry_points:
            raise ValueError(
                f"Pipeline {self.model_type} has no entry point "
                f"(stage with empty input_sources)"
            )

    def get_final_output_stages(self) -> list:
        return [s for s in self.stages if s.final_output]

    def get_stage(self, stage_id: int) -> OmniStageConfig:
        for s in self.stages:
            if s.stage_id == stage_id:
                return s
        raise KeyError(f"Stage {stage_id} not found in pipeline {self.model_type}")
