#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "autil/EnvUtil.h"
#include <stdexcept>

using namespace autil;

namespace rtp_llm {

EngineBase::EngineBase(const EngineInitParams& params) {
    initRuntime(params);
}

EngineBase::~EngineBase() {}

std::vector<GenerateStreamPtr> EngineBase::batchEnqueue(const std::vector<std::shared_ptr<GenerateInput>>& inputs) {
    throw std::runtime_error("not implemeted");
}

std::shared_ptr<GenerateStream> EngineBase::makeStream(const std::shared_ptr<GenerateInput>& input) {
    throw std::runtime_error("not implemeted");
}

void EngineBase::initRuntime(const EngineInitParams& params) {
    const auto rank =
        params.parallelism_config.dp_rank * params.parallelism_config.tp_size + params.parallelism_config.tp_rank;
    Logger::getEngineLogger().setRank(rank);
    Logger::getEngineLogger().flush();
    // Per-engine device id. The standard formula assigns one GPU per local rank.
    // For multi-stage models (e.g. Qwen Omni thinker+talker) running in one process
    // on different GPUs, the caller sets per-stage parallelism_config with distinct
    // world_rank values and local_world_size >= num_stages, so each engine gets a
    // distinct device_id and rtp_llm::initRuntime switches the current thread's
    // device before this engine's allocations happen.
    device_id_ = static_cast<int64_t>(
        params.parallelism_config.world_rank % params.parallelism_config.local_world_size);
    mla_ops_type_ = rtp_llm::initRuntime(static_cast<size_t>(device_id_),
                                         params.profiling_debug_logging_config.trace_memory,
                                         params.device_resource_config.enable_comm_overlap,
                                         params.model_config_.mla_ops_type);
    warmupNoBlockCopy();
}

std::shared_ptr<KVCacheManager> EngineBase::getCacheManager() const {
    return resource_context_.cache_manager;
}

}  // namespace rtp_llm
