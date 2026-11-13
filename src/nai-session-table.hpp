// nai-session-table.hpp
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <chrono>
#include <vector>

constexpr uint64_t NAI_GOLDEN_RATIO = 11400714819323198485ULL;
constexpr uint64_t NAI_TOMBSTONE_HANDLE = 0xFFFFFFFFFFFFFFFFULL;

struct alignas(64) NaiSession {
    uint64_t session_handle;
    uint64_t packed_user_id;
    int64_t  cliente_local_id;
    uint64_t name_hash;
    int64_t  last_activity;
    uint32_t device_hash;
    int32_t  sucursal_id;
    uint16_t bloque_id;
    uint16_t permissions_mask;
    uint8_t  _padding[20];
    
    NaiSession() { std::memset(this, 0, sizeof(NaiSession)); }
};