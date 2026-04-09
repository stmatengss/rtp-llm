#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeControlService.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace transfer {
namespace mooncake {

namespace {

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

}  // namespace

MooncakeControlService::MooncakeControlService(const std::shared_ptr<TransferTaskStore>& task_store,
                                               const MooncakeTransferEnginePtr&          engine):
    task_store_(task_store), engine_(engine) {}

void MooncakeControlService::prepare(::google::protobuf::RpcController*                 controller,
                                     const ::mooncake_transfer::MooncakePrepareRequest* request,
                                     ::mooncake_transfer::MooncakePrepareResponse*      response,
                                     ::google::protobuf::Closure*                       done) {
    (void)controller;
    auto done_guard = std::unique_ptr<::google::protobuf::Closure, void (*)(::google::protobuf::Closure*)>(
        done, [](::google::protobuf::Closure* closure) { if (closure) { closure->Run(); } });

    if (!request->has_unique_key() || request->unique_key().empty()) {
        response->set_error_code(::mooncake_transfer::MOONCAKE_TRANSFER_UNKNOWN_ERROR);
        response->set_error_message("prepare request missing unique_key");
        return;
    }

    auto task = task_store_->getTask(request->unique_key());
    if (!task) {
        auto error_code = (request->has_deadline_ms() && currentTimeMs() >= request->deadline_ms()) ?
                              TransferErrorCode::TIMEOUT :
                              TransferErrorCode::UNKNOWN;
        response->set_error_code(toProtoErrorCode(error_code));
        response->set_error_message("matching recv task not found");
        return;
    }

    if (task->done()) {
        response->set_error_code(toProtoErrorCode(task->errorCode()));
        response->set_error_message(task->errorMessage());
        return;
    }

    response->set_error_code(::mooncake_transfer::MOONCAKE_TRANSFER_NONE_ERROR);
    response->set_unique_key(request->unique_key());
    response->set_segment_name(engine_ ? engine_->localSegmentName() : std::string());
    for (const auto& [cache_key, key_block_info] : task->getBlockInfos()) {
        auto* remote_key_info = response->add_blocks();
        remote_key_info->set_key(cache_key);
        for (const auto& block_info : key_block_info->blocks) {
            if (block_info.addr == nullptr || block_info.size_bytes == 0) {
                continue;
            }
            auto* remote_block = remote_key_info->add_blocks();
            remote_block->set_remote_addr(reinterpret_cast<uint64_t>(block_info.addr));
            remote_block->set_len(static_cast<uint32_t>(block_info.size_bytes));
        }
    }
}

void MooncakeControlService::finish(::google::protobuf::RpcController*                controller,
                                    const ::mooncake_transfer::MooncakeFinishRequest* request,
                                    ::mooncake_transfer::MooncakeFinishResponse*      response,
                                    ::google::protobuf::Closure*                       done) {
    (void)controller;
    auto done_guard = std::unique_ptr<::google::protobuf::Closure, void (*)(::google::protobuf::Closure*)>(
        done, [](::google::protobuf::Closure* closure) { if (closure) { closure->Run(); } });

    if (!request->has_unique_key() || request->unique_key().empty()) {
        response->set_error_code(::mooncake_transfer::MOONCAKE_TRANSFER_UNKNOWN_ERROR);
        response->set_error_message("finish request missing unique_key");
        return;
    }

    auto task = task_store_->stealTask(request->unique_key());
    if (!task) {
        response->set_error_code(::mooncake_transfer::MOONCAKE_TRANSFER_NONE_ERROR);
        return;
    }

    const auto task_error = request->has_error_code() ? fromProtoErrorCode(request->error_code()) : TransferErrorCode::OK;
    const auto task_msg   = request->has_error_message() ? request->error_message() : std::string();
    task->notifyDone(task_error == TransferErrorCode::OK, task_error, task_msg);

    response->set_error_code(::mooncake_transfer::MOONCAKE_TRANSFER_NONE_ERROR);
}

}  // namespace mooncake
}  // namespace transfer
}  // namespace rtp_llm