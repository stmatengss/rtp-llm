# Multi-Model Omni Compatibility Design

## Problem

The current rtp-llm omni framework works for Qwen2.5-Omni but cannot support other omni models (GLM-4-Voice, MiniOmni, Moshi, Qwen3-Omni, MiniCPM-o, etc.) without significant refactoring. Key issues:

1. **OmniEngine** has no Qwen-specific code now (it's generic), but the talker runs through a pure-Python `TalkerEngineWrapper` instead of the rtp C++ engine
2. **StageProcessorBase** uses an ABC + registry class hierarchy — too heavy for simple inter-stage transforms
3. **OmniStageConfig** lacks fields needed by other models: `owns_tokenizer`, `model_subdir`, `sampling_constraints`, `custom_process_input_func`
4. **Weight loading** assumes a single shared checkpoint with prefix-based separation — doesn't support models with separate checkpoints per stage

## Constraint

**Both thinker and talker MUST use the rtp C++ engine.** If the C++ engine doesn't support a stage's requirements, modify the C++ code rather than falling back to Python.

## Reference

vllm-omni (https://github.com/vllm-project/vllm-omni) supports 25+ models with:
- `StagePipelineConfig` frozen dataclass with function-string stage processors
- Per-stage architecture registration in model registry
- YAML-based deployment configs (topology vs. runtime separation)
- `stage_input_processors/` directory with plain functions per model
- `StagePool` with ZMQ-based inter-process communication

## Design

### 1. Extended OmniStageConfig

Add fields to `OmniStageConfig` to support diverse model architectures:

```python
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

    # NEW: function-string for inter-stage transform (replaces StageProcessorBase ABC)
    custom_process_input_func: Optional[str] = None

    # NEW: which stage owns tokenizer (entry stage)
    owns_tokenizer: bool = False

    # NEW: per-stage checkpoint subdirectory for split-weight models
    model_subdir: Optional[str] = None

    # NEW: per-stage sampling constraints (stop tokens, temperature, etc.)
    sampling_constraints: Dict[str, Any] = field(default_factory=dict)

    # NEW: per-stage engine config overrides
    engine_overrides: Dict[str, Any] = field(default_factory=dict)
```

### 2. Remove StageProcessorBase ABC and StageProcessorRegistry

Replace with plain functions resolved via dotted-path strings:

```python
# Before: ABC class hierarchy
class ThinkerToTalkerProcessor(StageProcessorBase):
    def process(self, source_output: StageOutput) -> StageOutput: ...

# After: plain function
def thinker2talker(source_output: StageOutput, pipeline_config: OmniPipelineConfig) -> StageOutput:
    ...
```

The `custom_process_input_func` field in `OmniStageConfig` stores the dotted path (e.g., `"rtp_llm.omni.models.qwen2_5_omni.stage_processors.thinker2talker"`), resolved at runtime via `importlib.import_module()`.

### 3. C++ Engine Extensions for Talker Support

The talker currently runs through `TalkerEngineWrapper` which reimplements the entire transformer forward pass in Python (~270 lines). This must be replaced with C++ engine execution.

The talker's special requirement: `embed_tokens(codec_token) + thinker_hidden_state → thinker_to_talker_proj → decoder layers`.

#### Option A: `external_embeddings` input (Recommended)

Add a new input field `external_embeddings` to `GptModelInputs`:
- Shape: `[seq_len, embed_dim]`
- When present, replaces the standard `embed_tokens()` output
- The Python side pre-computes: `embed + thinker_hs → proj` and passes the result as `external_embeddings`
- The C++ embedding layer checks: if `external_embeddings` provided, use it; otherwise, do normal token embedding

This approach:
- Minimal C++ change (one if-check in the embedding path)
- Keeps projection logic in Python (model-specific, easy to customize per model)
- Reusable for any model that needs custom embedding (GLM-4-Voice, MiniOmni, etc.)

Required C++ changes:
1. Add `external_embeddings` field to `GptModelInputs` struct
2. Add `externalEmbeddings` / `externalEmbeddingsDtype` to `GptModelInputIndex`
3. In the embedding kernel: if `external_embeddings` is set, skip `embed_tokens()` and use it directly
4. Add `external_embeddings` parameter to pybind `generate()` method

#### Option B: `pre_decoder_projection` in C++

Add a configurable linear projection layer after embedding. More C++ code, less Python flexibility. Not recommended.

### 4. Multi-Engine OmniEngine

`OmniEngine` manages multiple `RtpLLMOp` instances (one per stage that uses the C++ engine). Each stage gets its own engine initialization with its own model config, weights, and KV cache.

```python
class OmniEngine:
    def __init__(self, pipeline_config, model_config, engine_config):
        self.stage_engines: Dict[int, RtpLLMOp] = {}
        for stage in pipeline_config.stages:
            if stage.execution_type in (StageExecutionType.LLM_AR, StageExecutionType.LLM_GENERATION):
                engine = self._create_stage_engine(stage, model_config, engine_config)
                self.stage_engines[stage.stage_id] = engine

    async def generate(self, request):
        for stage_id in self.orchestrator.get_execution_order():
            stage = self.pipeline_config.get_stage(stage_id)
            if stage.custom_process_input_func:
                transform = resolve_func(stage.custom_process_input_func)
                stage_input = transform(prev_output, self.pipeline_config)
            result = self.stage_engines[stage_id].generate(stage_input)
            prev_output = result
        return self.output_processor.process(prev_output)
```

### 5. Weight Loading for Multi-Stage Models

Three patterns supported:

1. **Shared checkpoint + prefix** (Qwen2.5-Omni): Single `config.json` with `thinker_config`, `talker_config` sub-sections. Weight prefix: `"thinker."`, `"talker."`. Already works.

2. **Separate checkpoints** (some models): Each stage has its own directory. Use `model_subdir` in `OmniStageConfig` to point to the subdirectory.

3. **Shared checkpoint + HF sub-config** (Qwen3-Omni, MiniCPM-o): Single checkpoint, but each stage reads a different `architectures` entry. Use `model_arch` per stage.

### 6. New Model Registration Pattern

Each model provides a `pipeline_config.py` file that:
1. Declares the `OmniPipelineConfig` with all stages
2. Registers it with `OmniPipelineRegistry`
3. Registers each stage's model class with `register_model()`

```python
# rtp_llm/omni/models/glm4_voice/pipeline_config.py
GLM4_VOICE_PIPELINE = OmniPipelineConfig(
    model_type="glm4_voice",
    model_arch="GLM4VoiceModel",
    stages=(
        OmniStageConfig(
            stage_id=0,
            model_stage="encoder",
            execution_type=StageExecutionType.LLM_AR,
            model_cls="glm4_voice_encoder",
            owns_tokenizer=True,
            final_output=True,
            final_output_type="text",
            engine_output_type="latent",
        ),
        OmniStageConfig(
            stage_id=1,
            model_stage="decoder",
            execution_type=StageExecutionType.LLM_AR,
            model_cls="glm4_voice_decoder",
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            custom_process_input_func="rtp_llm.omni.models.glm4_voice.stage_processors.encoder2decoder",
        ),
    ),
)
OmniPipelineRegistry.register(GLM4_VOICE_PIPELINE)
```

### 7. Files to Change

**Modify:**
- `rtp_llm/omni/config/stage_config.py` — add new fields
- `rtp_llm/omni/engine/omni_engine.py` — multi-engine support, function-string processor resolution
- `rtp_llm/omni/engine/orchestrator.py` — support non-linear topologies (DAG)
- `rtp_llm/omni/models/qwen2_5_omni/pipeline_config.py` — use new fields
- `rtp_llm/omni/models/qwen2_5_omni/stage_processors.py` — convert to plain functions
- `rtp_llm/cpp/models/ModelTypes.h` — add `externalEmbeddings` to `GptModelInputIndex`
- `rtp_llm/cpp/models/ModelTypes.cc` — handle external embeddings sync
- C++ embedding kernel — check for external embeddings

**Remove:**
- `rtp_llm/omni/engine/stage_processor_base.py` — replaced by plain functions
- `rtp_llm/omni/engine/stage_processor_registry.py` — no longer needed
- `rtp_llm/omni/models/qwen2_5_omni/talker_engine_wrapper.py` — replaced by C++ engine

**Add:**
- `rtp_llm/omni/engine/func_resolver.py` — utility to resolve dotted-path function strings

## Scope

Phase 1 (this PR):
1. Extend `OmniStageConfig` with new fields
2. Replace `StageProcessorBase`/Registry with function-string resolution
3. C++ `external_embeddings` support
4. Refactor talker to use C++ engine with `external_embeddings`
5. Update Qwen2.5-Omni pipeline config
6. Unit tests

Phase 2 (follow-up):
- Add GLM-4-Voice support as second model
- Add Qwen3-Omni support
- Streaming output support
- Multi-GPU stage placement
