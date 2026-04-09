#pragma once

#include <memory>

#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeTransferEngine.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/proto/mooncake_service.pb.h"

namespace rtp_llm {
namespace transfer {
namespace mooncake {

class MooncakeControlService: public ::mooncake_transfer::MooncakeTransferControlService {
public:
    MooncakeControlService(const std::shared_ptr<TransferTaskStore>& task_store,
                           const MooncakeTransferEnginePtr&          engine);
    ~MooncakeControlService() override = default;

public:
    void prepare(::google::protobuf::RpcController*                        controller,
                 const ::mooncake_transfer::MooncakePrepareRequest*        request,
                 ::mooncake_transfer::MooncakePrepareResponse*             response,
                 ::google::protobuf::Closure*                              done) override;

    void finish(::google::protobuf::RpcController*                         controller,
                const ::mooncake_transfer::MooncakeFinishRequest*          request,
                ::mooncake_transfer::MooncakeFinishResponse*               response,
                ::google::protobuf::Closure*                               done) override;

private:
    std::shared_ptr<TransferTaskStore> task_store_;
    MooncakeTransferEnginePtr          engine_;
};

}  // namespace mooncake
}  // namespace transfer
}  // namespace rtp_llm