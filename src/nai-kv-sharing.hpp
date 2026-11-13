#ifndef NAI_KV_SHARING_HPP
#define NAI_KV_SHARING_HPP

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <array>
#include <mutex>

namespace nai {


constexpr uint32_t NAI_FNV_OFFSET_BASIS = 2166136261u;
constexpr uint32_t NAI_FNV_PRIME        = 16777619u;

inline uint32_t nai_hash_prefix(const int32_t * tokens, size_t n_tokens, size_t max_tokens = 512) {
    uint32_t hash = NAI_FNV_OFFSET_BASIS;
    size_t limit = (n_tokens < max_tokens) ? n_tokens : max_tokens;
    
    for (size_t i = 0; i < limit; ++i) {
        hash ^= static_cast<uint32_t>(tokens[i]);
        hash *= NAI_FNV_PRIME;
    }
    return hash;
}

class NaiKvSharing {
private:
    struct Entry {
        uint32_t hash;
        int32_t  seq_id;
        bool     valid;
    };

    static constexpr size_t SHARD_COUNT = 64;
    static constexpr size_t MAX_ENTRIES = 8192;

    std::array<Entry, MAX_ENTRIES> _entries;
    
    mutable std::array<std::mutex, SHARD_COUNT> _locks;

public:
    NaiKvSharing() {
        clear();
    }

    void register_prefix(uint32_t hash, int32_t seq_id) {
        size_t idx = hash % MAX_ENTRIES;
        std::lock_guard<std::mutex> lock(_locks[idx % SHARD_COUNT]);
        
        if (!_entries[idx].valid) {
            _entries[idx].hash   = hash;
            _entries[idx].seq_id = seq_id;
            _entries[idx].valid  = true;
        }
    }

    bool try_get_source(uint32_t hash, int32_t& out_seq_id) const {
        size_t idx = hash % MAX_ENTRIES;
        std::lock_guard<std::mutex> lock(_locks[idx % SHARD_COUNT]);
        
        if (_entries[idx].valid && _entries[idx].hash == hash) {
            out_seq_id = _entries[idx].seq_id;
            return true;
        }
        return false;
    }

    void clear() {
        for (size_t i = 0; i < SHARD_COUNT; ++i) {
            std::lock_guard<std::mutex> lock(_locks[i]);
        }
        for (auto& entry : _entries) {
            entry.valid = false;
            entry.hash = 0;
            entry.seq_id = -1;
        }
    }
};

extern NaiKvSharing g_nai_kv_sharing;

}

#endif