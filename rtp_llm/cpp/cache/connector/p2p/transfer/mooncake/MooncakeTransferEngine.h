#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>

#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheSender.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendConfig.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeTypes.h"

namespace rtp_llm {
namespace transfer {
namespace mooncake {

class MooncakeTransferEngine {
public:
    explicit MooncakeTransferEngine(const TransferBackendConfig& config);
    ~MooncakeTransferEngine() = default;

public:
    bool init(bool receiver_side);
    bool available() const;
    bool regMem(const BlockInfo& block_info, uint64_t aligned_size, std::string* error_message = nullptr);

    const std::string& localSegmentName() const {
        return local_segment_name_;
    }

    bool submitWrite(const SendRequest&                 request,
                     const MooncakeRemoteDescriptor&    descriptor,
                     std::string*                       error_message = nullptr);

private:
    TransferBackendConfig config_;
    std::string           local_segment_name_;
    bool                  initialized_ = false;
    mutable std::mutex    mutex_;
    std::unordered_set<void*> registered_addrs_;
};

using MooncakeTransferEnginePtr = std::shared_ptr<MooncakeTransferEngine>;

}  // namespace mooncake
}  // namespace transfer
}  // namespace rtp_llm