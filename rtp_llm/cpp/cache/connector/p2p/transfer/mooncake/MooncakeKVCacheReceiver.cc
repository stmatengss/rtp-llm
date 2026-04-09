#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeKVCacheReceiver.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace transfer {
namespace mooncake {

MooncakeKVCacheReceiver::MooncakeKVCacheReceiver(const TransferBackendConfig&        config,
                                                 const kmonitor::MetricsReporterPtr& metrics_reporter):
    config_(config), task_store_(std::make_shared<TransferTaskStore>()), metrics_reporter_(metrics_reporter) {}

MooncakeKVCacheReceiver::~MooncakeKVCacheReceiver() {
    tcp_server_.reset();
    control_service_.reset();
    engine_.reset();
}

bool MooncakeKVCacheReceiver::init(uint32_t listen_port,
                                   int      io_thread_count,
                                   int      worker_thread_count,
                                   uint32_t anet_rpc_thread_num,
                                   uint32_t anet_rpc_queue_num) {
    engine_ = std::make_shared<MooncakeTransferEngine>(config_);
    if (!engine_->init(true)) {
        RTP_LLM_LOG_WARNING("MooncakeKVCacheReceiver init failed: engine init failed");
        return false;
    }

    control_service_ = std::make_shared<MooncakeControlService>(task_store_, engine_);
    tcp_server_      = std::make_shared<transfer::TcpServer>();
    if (!tcp_server_->init(io_thread_count,
                           worker_thread_count,
                           listen_port,
                           true,
                           anet_rpc_thread_num,
                           anet_rpc_queue_num)) {
        RTP_LLM_LOG_WARNING("MooncakeKVCacheReceiver init failed: create tcp server failed");
        return false;
    }
    if (!tcp_server_->registerService(control_service_.get())) {
        RTP_LLM_LOG_WARNING("MooncakeKVCacheReceiver init failed: register control service failed");
        return false;
    }
    if (!tcp_server_->start()) {
        RTP_LLM_LOG_WARNING("MooncakeKVCacheReceiver init failed: start tcp server failed");
        return false;
    }
    return true;
}

bool MooncakeKVCacheReceiver::regMem(const BlockInfo& block_info, uint64_t aligned_size) {
    std::string error_message;
    const bool  ok = engine_ && engine_->regMem(block_info, aligned_size, &error_message);
    if (!ok) {
        RTP_LLM_LOG_WARNING("MooncakeKVCacheReceiver regMem failed: %s", error_message.c_str());
    }
    return ok;
}

transfer::IKVCacheRecvTaskPtr MooncakeKVCacheReceiver::recv(const transfer::RecvRequest& request) {
    return task_store_->addTask(request.unique_key, request.block_info, request.deadline_ms);
}

void MooncakeKVCacheReceiver::stealTask(const std::string& unique_key) {
    task_store_->stealTask(unique_key);
}

transfer::IKVCacheRecvTaskPtr MooncakeKVCacheReceiver::getTask(const std::string& unique_key) {
    return task_store_->getTask(unique_key);
}

}  // namespace mooncake
}  // namespace transfer
}  // namespace rtp_llm