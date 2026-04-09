package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mooncake_types",
    hdrs = ["MooncakeTypes.h"],
    deps = [
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
    ],
)

cc_library(
    name = "mooncake_engine",
    hdrs = ["MooncakeTransferEngine.h"],
    srcs = ["MooncakeTransferEngine.cc"],
    deps = [
        ":mooncake_types",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:transfer_backend_config",
        "//rtp_llm/cpp/utils:core_utils",
    ],
)

cc_library(
    name = "mooncake_control_service",
    hdrs = ["MooncakeControlService.h"],
    srcs = ["MooncakeControlService.cc"],
    deps = [
        ":mooncake_engine",
        ":mooncake_types",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "//rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/proto:mooncake_service_cc_proto",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:time_util",
    ],
)

cc_library(
    name = "mooncake_backend",
    hdrs = [
        "MooncakeKVCacheReceiver.h",
        "MooncakeKVCacheSender.h",
    ],
    srcs = [
        "MooncakeKVCacheReceiver.cc",
        "MooncakeKVCacheSender.cc",
    ],
    deps = [
        ":mooncake_control_service",
        ":mooncake_engine",
        ":mooncake_types",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:arpc_dep",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:transfer_metric",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:transfer_backend_config",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "//rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/proto:mooncake_service_cc_proto",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:time_util",
        "@havenask//aios/network/arpc:arpc",
    ],
)