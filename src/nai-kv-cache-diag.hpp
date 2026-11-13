#ifndef NAI_KV_CACHE_DIAG_HPP
#define NAI_KV_CACHE_DIAG_HPP

#include <atomic>
#include <cstdint>
#include <cstdio>

namespace nai::kv_diag {

#ifdef LLAMA_NAI_TELEMETRY

inline std::atomic<uint64_t> g_clears     {0};
inline std::atomic<uint64_t> g_slot_fails {0};
inline std::atomic<uint64_t> g_ubatches   {0};

inline std::atomic<double>   g_last_fill_ratio {-1.0};
inline std::atomic<uint32_t> g_last_used  {0};
inline std::atomic<uint32_t> g_last_size  {0};

inline void trace_clear(bool /*data*/) {
    g_clears.fetch_add(1, std::memory_order_relaxed);
}

inline void trace_slot_fail(
        int32_t /*seq_id*/,
        uint32_t /*n_tokens*/,
        uint32_t used,
        uint32_t size,
        uint32_t /*head*/) {
    g_slot_fails.fetch_add(1, std::memory_order_relaxed);

    // guardar la última observación de fill ratio
    const double fr = size ? double(used) / double(size) : 0.0;
    g_last_fill_ratio.store(fr,          std::memory_order_relaxed);
    g_last_used.store      (used,        std::memory_order_relaxed);
    g_last_size.store      (size,        std::memory_order_relaxed);
}

inline void trace_ubatch_applied(
        uint32_t /*n_tokens*/,
        uint32_t /*n_streams*/,
        uint32_t /*head_after*/) {
    g_ubatches.fetch_add(1, std::memory_order_relaxed);
}

inline void dump() {
    const uint64_t c  = g_clears    .load(std::memory_order_relaxed);
    const uint64_t f  = g_slot_fails.load(std::memory_order_relaxed);
    const uint64_t u  = g_ubatches  .load(std::memory_order_relaxed);
    const double   r  = g_last_fill_ratio.load(std::memory_order_relaxed);
    const uint32_t lu = g_last_used .load(std::memory_order_relaxed);
    const uint32_t ls = g_last_size .load(std::memory_order_relaxed);

    std::fprintf(stderr,
        "[nai-kv] clears=%llu slot_fails=%llu ubatches=%llu last_fill=%.3f (%u/%u)\n",
        (unsigned long long)c,
        (unsigned long long)f,
        (unsigned long long)u,
        r, lu, ls);

    if (f == 0) {
        std::fprintf(stderr,
            "[nai-kv] -> 0 failures. Fase 3.3 (eviction gravitacional) NO se justifica en este uso.\n");
    } else if (r < 0.0 || r > 1.0) {
        std::fprintf(stderr,
            "[nai-kv] -> %llu failures con fill desconocido. Revisar trazabilidad.\n",
            (unsigned long long)f);
    } else if (r >= 0.95) {
        std::fprintf(stderr,
            "[nai-kv] -> %llu failures a fill>=0.95. Candidato claro para Fase 3.3.\n",
            (unsigned long long)f);
    } else {
        std::fprintf(stderr,
            "[nai-kv] -> %llu failures con fill %.3f. Revisar distribucion de masas antes de Fase 3.3.\n",
            (unsigned long long)f, r);
    }
}

#else

inline void trace_clear(bool)                                              {}
inline void trace_slot_fail(int32_t, uint32_t, uint32_t, uint32_t, uint32_t) {}
inline void trace_ubatch_applied(uint32_t, uint32_t, uint32_t)             {}
inline void dump()                                                          {}

#endif

}

#define NAI_KV_TRACE_CLEAR(data) \
    ::nai::kv_diag::trace_clear(data)

#define NAI_KV_TRACE_SLOT_FAIL(seq_id, n_tokens, used, size, head) \
    ::nai::kv_diag::trace_slot_fail(seq_id, n_tokens, used, size, head)

#define NAI_KV_TRACE_UBATCH_APPLIED(n_tokens, n_streams, head_after) \
    ::nai::kv_diag::trace_ubatch_applied(n_tokens, n_streams, head_after)

#endif // NAI_KV_CACHE_DIAG_HPP