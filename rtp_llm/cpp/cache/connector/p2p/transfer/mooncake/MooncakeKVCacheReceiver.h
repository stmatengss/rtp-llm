#pragma once

#include <memory>

#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheReceiver.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TcpServer.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendConfig.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeControlService.h"

namespace rtp_llm {
namespace transfer {
namespace mooncake {

class MooncakeKVCacheReceiver: public transfer::IKVCacheReceiver {
public:
    MooncakeKVCacheReceiver(const TransferBackendConfig&          config,
                            const kmonitor::MetricsReporterPtr&   metrics_reporter = nullptr);
    ~MooncakeKVCacheReceiver() override;

public:
    bool init(uint32_t listen_port,
              int      io_thread_count,
              int      worker_thread_count,
              uint32_t anet_rpc_thread_num = 3,
              uint32_t anet_rpc_queue_num  = 100);

    bool regMem(const BlockInfo& block_info, uint64_t aligned_size = 0) override;
    transfer::IKVCacheRecvTaskPtr recv(const transfer::RecvRequest& request) override;

    void                          stealTask(const std::string& unique_key) override;
    transfer::IKVCacheRecvTaskPtr getTask(const std::string& unique_key) override;

private:
    TransferBackendConfig               config_;
    std::shared_ptr<transfer::TcpServer> tcp_server_;
    std::shared_ptr<TransferTaskStore>   task_store_;
    std::shared_ptr<MooncakeControlService> control_service_;
    MooncakeTransferEnginePtr              engine_;
    kmonitor::MetricsReporterPtr           metrics_reporter_;
};

}  // namespace mooncake
}  // namespace transfer
}  // namespace rtp_llm