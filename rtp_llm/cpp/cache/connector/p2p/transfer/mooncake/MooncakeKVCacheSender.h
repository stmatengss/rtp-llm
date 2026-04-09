#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>

#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheSender.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TcpClient.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendConfig.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferMetric.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeTransferEngine.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeTypes.h"

namespace rtp_llm {
namespace transfer {
namespace mooncake {

class MooncakeKVCacheSender: public transfer::IKVCacheSender {
public:
    MooncakeKVCacheSender(const TransferBackendConfig&        config,
                          const kmonitor::MetricsReporterPtr& metrics_reporter = nullptr);
    ~MooncakeKVCacheSender() override = default;

public:
    bool init(int                       io_thread_count,
              std::chrono::milliseconds channel_idle_ttl     = std::chrono::milliseconds::zero(),
              std::uint64_t             sweep_interval_calls = 0);

    bool regMem(const BlockInfo& block_info, uint64_t aligned_size = 0) override;

    void send(const transfer::SendRequest&                               request,
              std::function<void(TransferErrorCode, const std::string&)> callback) override;

    // Internal helpers used by async RPC closures.
    bool validateDescriptor(const SendRequest& request, const MooncakeRemoteDescriptor& descriptor, std::string* error_message);

    void finishRemote(const std::string&                                         ip,
                      uint32_t                                                   port,
                      const std::string&                                         unique_key,
                      TransferErrorCode                                          result_code,
                      const std::string&                                         result_message,
                      std::function<void(TransferErrorCode, const std::string&)> callback);

    const MooncakeTransferEnginePtr& engine() const {
        return engine_;
    }

private:
    TransferBackendConfig              config_;
    std::shared_ptr<transfer::TcpClient> tcp_client_;
    MooncakeTransferEnginePtr            engine_;
    kmonitor::MetricsReporterPtr         metrics_reporter_;
};

}  // namespace mooncake
}  // namespace transfer
}  // namespace rtp_llm