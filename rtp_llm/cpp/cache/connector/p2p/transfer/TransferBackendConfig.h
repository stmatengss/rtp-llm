#pragma once

#include <cstdint>
#include <string>

namespace rtp_llm {
namespace transfer {

struct TransferBackendConfig {
    bool    cache_store_rdma_mode               = false;
    int64_t rdma_transfer_wait_timeout_ms       = 180 * 1000;
    int     messager_io_thread_count            = 2;
    int     messager_worker_thread_count        = 16;
    int     rdma_max_block_pairs_per_connection = 0;
    int64_t cache_store_listen_port             = 0;
    int     cache_store_tcp_anet_rpc_thread_num = 3;
    int     cache_store_tcp_anet_rpc_queue_num  = 100;
    /// 0: 关闭 TcpClient channel idle 淘汰；大于 0 为毫秒
    int64_t tcp_channel_idle_ttl_ms = 0;
    /// 0: 关闭每 N 次 getChannel 的全表清扫，仅 miss 时清扫；大于 0 为间隔
    int64_t tcp_channel_sweep_interval_calls = 0;
    /// Tcp/Rdma TransferService::waitCheckProc 轮询间隔（微秒）；<=0 时实现侧按 1000（1ms）处理
    int64_t transfer_wait_check_interval_us = 1000;

    /// Mooncake classic TransferEngine metadata 连接串，例如 etcd://127.0.0.1:2379
    std::string mooncake_metadata_conn_string;
    /// Mooncake 本地 segment/server 名称；为空时实现侧会回退到 ip:port 形式
    std::string mooncake_local_server_name;
    /// Mooncake init 使用的本地 ip/hostname；为空时由实现自行推断
    std::string mooncake_local_ip_or_host_name;
    /// Mooncake classic TransferEngine 的 RPC 端口
    int64_t mooncake_rpc_port = 12345;
    /// Mooncake registerLocalMemory 的 location 参数
    std::string mooncake_location = "*";
    /// 是否启用 Mooncake 自动发现
    bool mooncake_auto_discover = false;
};

}  // namespace transfer
}  // namespace rtp_llm
