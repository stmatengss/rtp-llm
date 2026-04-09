#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeKVCacheSender.h"

#include <memory>

#include "aios/network/arpc/arpc/ANetRPCController.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/proto/mooncake_service.pb.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace transfer {
namespace mooncake {

namespace {

TransferErrorCode fromProtoErrorCode(::mooncake_transfer::MooncakeTransferErrorCodePB ec) {
    switch (ec) {
        case ::mooncake_transfer::MOONCAKE_TRANSFER_NONE_ERROR:
            return TransferErrorCode::OK;
        case ::mooncake_transfer::MOONCAKE_TRANSFER_TIMEOUT:
            return TransferErrorCode::TIMEOUT;
        case ::mooncake_transfer::MOONCAKE_TRANSFER_CANCELLED:
            return TransferErrorCode::CANCELLED;
        case ::mooncake_transfer::MOONCAKE_TRANSFER_BUFFER_MISMATCH:
            return TransferErrorCode::BUFFER_MISMATCH;
        default:
            return TransferErrorCode::UNKNOWN;
    }
}

::mooncake_transfer::MooncakeTransferErrorCodePB toProtoErrorCode(TransferErrorCode ec) {
    switch (ec) {
        case TransferErrorCode::OK:
            return ::mooncake_transfer::MOONCAKE_TRANSFER_NONE_ERROR;
        case TransferErrorCode::TIMEOUT:
            return ::mooncake_transfer::MOONCAKE_TRANSFER_TIMEOUT;
        case TransferErrorCode::CANCELLED:
            return ::mooncake_transfer::MOONCAKE_TRANSFER_CANCELLED;
        case TransferErrorCode::BUFFER_MISMATCH:
            return ::mooncake_transfer::MOONCAKE_TRANSFER_BUFFER_MISMATCH;
        default:
            return ::mooncake_transfer::MOONCAKE_TRANSFER_UNKNOWN_ERROR;
    }
}

MooncakeRemoteDescriptor parseDescriptor(const ::mooncake_transfer::MooncakePrepareResponse& response) {
    MooncakeRemoteDescriptor descriptor;
    descriptor.unique_key  = response.has_unique_key() ? response.unique_key() : std::string();
    descriptor.segment_name = response.has_segment_name() ? response.segment_name() : std::string();
    descriptor.block_infos.reserve(static_cast<size_t>(response.blocks_size()));
    for (const auto& block_group : response.blocks()) {
        MooncakeRemoteKeyBlockInfo key_info;
        key_info.cache_key = block_group.key();
        key_info.blocks.reserve(static_cast<size_t>(block_group.blocks_size()));
        for (const auto& block : block_group.blocks()) {
            key_info.blocks.push_back(MooncakeRemoteBlockInfo{block.remote_addr(), block.len()});
        }
        descriptor.block_infos.emplace_back(std::move(key_info));
    }
    return descriptor;
}

class MooncakePrepareClosure: public ::google::protobuf::Closure {
public:
    MooncakePrepareClosure(MooncakeKVCacheSender*                                        sender,
                           transfer::SendRequest                                         request,
                           const std::shared_ptr<::mooncake_transfer::MooncakePrepareResponse>& response,
                           arpc::ANetRPCController*                                       controller,
                           std::function<void(TransferErrorCode, const std::string&)>     callback):
        sender_(sender),
        request_(std::move(request)),
        response_(response),
        controller_(controller),
        callback_(std::move(callback)) {}

    ~MooncakePrepareClosure() override {
        delete controller_;
    }

    void Run() override {
        if (controller_->Failed()) {
            callback_(TransferErrorCode::RPC_FAILED, "mooncake prepare failed: " + controller_->ErrorText());
            return;
        }
        const auto response_code = response_->has_error_code() ? fromProtoErrorCode(response_->error_code()) : TransferErrorCode::UNKNOWN;
        if (response_code != TransferErrorCode::OK) {
            callback_(response_code, response_->has_error_message() ? response_->error_message() : std::string());
            return;
        }

        auto descriptor = parseDescriptor(*response_);
        std::string error_message;
        if (!sender_->validateDescriptor(request_, descriptor, &error_message)) {
            sender_->finishRemote(request_.ip,
                                  request_.port,
                                  request_.unique_key,
                                  TransferErrorCode::BUFFER_MISMATCH,
                                  error_message,
                                  callback_);
            return;
        }

        const bool transfer_ok = sender_->engine()->submitWrite(request_, descriptor, &error_message);
        const auto transfer_ec = transfer_ok ? TransferErrorCode::OK : TransferErrorCode::UNKNOWN;
        sender_->finishRemote(request_.ip,
                              request_.port,
                              request_.unique_key,
                              transfer_ec,
                              error_message,
                              callback_);
    }

private:
    MooncakeKVCacheSender*                                        sender_;
    transfer::SendRequest                                         request_;
    std::shared_ptr<::mooncake_transfer::MooncakePrepareResponse> response_;
    arpc::ANetRPCController*                                      controller_;
    std::function<void(TransferErrorCode, const std::string&)>    callback_;
};

class MooncakeFinishClosure: public ::google::protobuf::Closure {
public:
    MooncakeFinishClosure(arpc::ANetRPCController*                                   controller,
                          std::function<void(TransferErrorCode, const std::string&)> callback,
                          TransferErrorCode                                           result_code,
                          std::string                                                 result_message):
        controller_(controller),
        callback_(std::move(callback)),
        result_code_(result_code),
        result_message_(std::move(result_message)) {}

    ~MooncakeFinishClosure() override {
        delete controller_;
    }

    void Run() override {
        if (controller_->Failed()) {
            RTP_LLM_LOG_WARNING("mooncake finish failed: %s", controller_->ErrorText().c_str());
        }
        callback_(result_code_, result_message_);
    }

private:
    arpc::ANetRPCController*                                   controller_;
    std::function<void(TransferErrorCode, const std::string&)> callback_;
    TransferErrorCode                                           result_code_;
    std::string                                                 result_message_;
};

}  // namespace

MooncakeKVCacheSender::MooncakeKVCacheSender(const TransferBackendConfig&        config,
                                             const kmonitor::MetricsReporterPtr& metrics_reporter):
    config_(config), metrics_reporter_(metrics_reporter) {}

bool MooncakeKVCacheSender::init(int                       io_thread_count,
                                 std::chrono::milliseconds channel_idle_ttl,
                                 std::uint64_t             sweep_interval_calls) {
    tcp_client_ = std::make_shared<transfer::TcpClient>();
    if (!tcp_client_->init(io_thread_count, channel_idle_ttl, sweep_interval_calls)) {
        RTP_LLM_LOG_WARNING("MooncakeKVCacheSender: create tcp client failed");
        return false;
    }

    engine_ = std::make_shared<MooncakeTransferEngine>(config_);
    if (!engine_->init(false)) {
        RTP_LLM_LOG_WARNING("MooncakeKVCacheSender: engine init failed");
        return false;
    }
    return true;
}

bool MooncakeKVCacheSender::regMem(const BlockInfo& block_info, uint64_t aligned_size) {
    std::string error_message;
    const bool  ok = engine_ && engine_->regMem(block_info, aligned_size, &error_message);
    if (!ok) {
        RTP_LLM_LOG_WARNING("MooncakeKVCacheSender regMem failed: %s", error_message.c_str());
    }
    return ok;
}

bool MooncakeKVCacheSender::validateDescriptor(const SendRequest&               request,
                                               const MooncakeRemoteDescriptor& descriptor,
                                               std::string*                    error_message) {
    if (descriptor.block_infos.size() != request.block_info.size()) {
        if (error_message) {
            *error_message = "remote descriptor block group count mismatch";
        }
        return false;
    }

    for (const auto& remote_key_info : descriptor.block_infos) {
        auto it = request.block_info.find(remote_key_info.cache_key);
        if (it == request.block_info.end()) {
            if (error_message) {
                *error_message = "remote descriptor contains unknown cache_key";
            }
            return false;
        }

        size_t non_empty_blocks = 0;
        for (const auto& block_info : it->second->blocks) {
            if (block_info.addr != nullptr && block_info.size_bytes > 0) {
                ++non_empty_blocks;
            }
        }
        if (non_empty_blocks != remote_key_info.blocks.size()) {
            if (error_message) {
                *error_message = "remote descriptor sub-block count mismatch";
            }
            return false;
        }

        size_t remote_index = 0;
        for (const auto& block_info : it->second->blocks) {
            if (block_info.addr == nullptr || block_info.size_bytes == 0) {
                continue;
            }
            if (remote_key_info.blocks[remote_index].len != block_info.size_bytes) {
                if (error_message) {
                    *error_message = "remote descriptor sub-block size mismatch";
                }
                return false;
            }
            ++remote_index;
        }
    }
    return true;
}

void MooncakeKVCacheSender::finishRemote(const std::string&                                         ip,
                                         uint32_t                                                   port,
                                         const std::string&                                         unique_key,
                                         TransferErrorCode                                          result_code,
                                         const std::string&                                         result_message,
                                         std::function<void(TransferErrorCode, const std::string&)> callback) {
    auto channel = tcp_client_->getChannel(ip, port);
    if (!channel) {
        callback(result_code, result_message);
        return;
    }

    auto request  = std::make_shared<::mooncake_transfer::MooncakeFinishRequest>();
    auto response = std::make_shared<::mooncake_transfer::MooncakeFinishResponse>();
    request->set_unique_key(unique_key);
    request->set_error_code(toProtoErrorCode(result_code));
    if (!result_message.empty()) {
        request->set_error_message(result_message);
    }

    auto* controller = new arpc::ANetRPCController();
    controller->SetExpireTime(1000);
    auto* closure = new MooncakeFinishClosure(controller, std::move(callback), result_code, result_message);

    ::mooncake_transfer::MooncakeTransferControlService_Stub stub((::google::protobuf::RpcChannel*)(channel.get()),
                                                                  ::google::protobuf::Service::STUB_DOESNT_OWN_CHANNEL);
    stub.finish(controller, request.get(), response.get(), closure);
}

void MooncakeKVCacheSender::send(const transfer::SendRequest&                               request,
                                 std::function<void(TransferErrorCode, const std::string&)> callback) {
    auto collector     = std::make_shared<TransferClientMetricsCollector>();
    auto start_time_us = currentTimeUs();
    auto callback2     = [callback = std::move(callback), collector, start_time_us, metrics_reporter = metrics_reporter_](
                         TransferErrorCode error_code, const std::string& error_msg) {
        collector->success    = (error_code == TransferErrorCode::OK);
        collector->latency_us = currentTimeUs() - start_time_us;
        if (metrics_reporter) {
            metrics_reporter->report<TransferMetric, TransferClientMetricsCollector>(nullptr, collector.get());
        }
        callback(error_code, error_msg);
    };

    for (const auto& [cache_key, key_block_info] : request.block_info) {
        (void)cache_key;
        for (const auto& block_info : key_block_info->blocks) {
            if (block_info.addr != nullptr && block_info.size_bytes > 0) {
                ++collector->block_count;
                collector->total_block_size += block_info.size_bytes;
            }
        }
    }

    auto channel = tcp_client_->getChannel(request.ip, request.port);
    if (!channel) {
        callback2(TransferErrorCode::CONNECTION_FAILED, "get control channel failed");
        return;
    }

    auto prepare_request  = std::make_shared<::mooncake_transfer::MooncakePrepareRequest>();
    auto prepare_response = std::make_shared<::mooncake_transfer::MooncakePrepareResponse>();
    prepare_request->set_unique_key(request.unique_key);
    prepare_request->set_deadline_ms(request.deadline_ms);

    auto* controller = new arpc::ANetRPCController();
    const auto timeout_ms = std::max<int64_t>(1, request.deadline_ms - currentTimeMs());
    controller->SetExpireTime(timeout_ms);
    auto* closure = new MooncakePrepareClosure(this, request, prepare_response, controller, std::move(callback2));

    ::mooncake_transfer::MooncakeTransferControlService_Stub stub((::google::protobuf::RpcChannel*)(channel.get()),
                                                                  ::google::protobuf::Service::STUB_DOESNT_OWN_CHANNEL);
    stub.prepare(controller, prepare_request.get(), prepare_response.get(), closure);
}

}  // namespace mooncake
}  // namespace transfer
}  // namespace rtp_llm