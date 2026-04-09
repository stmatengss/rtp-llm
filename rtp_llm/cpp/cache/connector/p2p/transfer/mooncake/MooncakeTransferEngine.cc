#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeTransferEngine.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace transfer {
namespace mooncake {

MooncakeTransferEngine::MooncakeTransferEngine(const TransferBackendConfig& config): config_(config) {}

bool MooncakeTransferEngine::init(bool /*receiver_side*/) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) {
        return true;
    }
    if (!config_.mooncake_local_server_name.empty()) {
        local_segment_name_ = config_.mooncake_local_server_name;
    } else if (!config_.mooncake_local_ip_or_host_name.empty()) {
        local_segment_name_ = config_.mooncake_local_ip_or_host_name + ":" + std::to_string(config_.mooncake_rpc_port);
    } else {
        local_segment_name_ = "mooncake-segment-" + std::to_string(config_.mooncake_rpc_port);
    }
    initialized_ = true;
    RTP_LLM_LOG_WARNING("MooncakeTransferEngine stub initialized without Mooncake SDK; data plane is not active");
    return true;
}

bool MooncakeTransferEngine::available() const {
    return false;
}

bool MooncakeTransferEngine::regMem(const BlockInfo& block_info, uint64_t /*aligned_size*/, std::string* error_message) {
    if (block_info.addr == nullptr || block_info.size_bytes == 0) {
        if (error_message) {
            *error_message = "invalid block info for Mooncake registration";
        }
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    registered_addrs_.insert(block_info.addr);
    return true;
}

bool MooncakeTransferEngine::submitWrite(const SendRequest&              request,
                                         const MooncakeRemoteDescriptor& descriptor,
                                         std::string*                    error_message) {
    (void)request;
    (void)descriptor;
    if (error_message) {
        *error_message = "Mooncake classic data plane is not linked in this build yet";
    }
    return false;
}

}  // namespace mooncake
}  // namespace transfer
}  // namespace rtp_llm