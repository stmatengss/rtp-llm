#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rtp_llm {
namespace transfer {
namespace mooncake {

struct MooncakeRemoteBlockInfo {
    uint64_t remote_addr = 0;
    uint32_t len         = 0;
};

struct MooncakeRemoteKeyBlockInfo {
    int64_t                             cache_key = 0;
    std::vector<MooncakeRemoteBlockInfo> blocks;
};

struct MooncakeRemoteDescriptor {
    std::string                           unique_key;
    std::string                           segment_name;
    std::vector<MooncakeRemoteKeyBlockInfo> block_infos;
};

}  // namespace mooncake
}  // namespace transfer
}  // namespace rtp_llm