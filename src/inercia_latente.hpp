#ifndef NAI_INERCIA_LATENTE_HPP
#define NAI_INERCIA_LATENTE_HPP

#include <cmath>
#include <cstdio>
#include <cstddef>
#include <vector>
#include <memory>
#include <atomic>
#include "ggml.h"

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
    #define NAI_CPU_PAUSE() _mm_pause()
#elif defined(__aarch64__)
    #define NAI_CPU_PAUSE() asm volatile("yield" ::: "memory")
#else
    #define NAI_CPU_PAUSE() do { } while (0)
#endif

struct InerciaLatenteResult {
    bool  es_autopista;
    float varianza;
    float entropia;
};

namespace nai {

class NaiSpinlock {
private:
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
public:
    void lock() noexcept {
        while (flag.test_and_set(std::memory_order_acquire)) {
            NAI_CPU_PAUSE();
        }
    }
    void unlock() noexcept {
        flag.clear(std::memory_order_release);
    }
    NaiSpinlock() = default;
    NaiSpinlock(const NaiSpinlock&) = delete;
    NaiSpinlock& operator=(const NaiSpinlock&) = delete;
};

class NaiScopedLock {
    NaiSpinlock& _l;
public:
    explicit NaiScopedLock(NaiSpinlock& l) noexcept : _l(l) { _l.lock(); }
    ~NaiScopedLock() { _l.unlock(); }
    NaiScopedLock(const NaiScopedLock&) = delete;
    NaiScopedLock& operator=(const NaiScopedLock&) = delete;
};

class MapaMasaSemantica {
private:
    static constexpr size_t SHARD_COUNT      = 64;
    static constexpr size_t DEFAULT_MAX_POS  = 65536;

    std::unique_ptr<std::atomic<float>[]> _masas;
    size_t _capacity;

    NaiSpinlock _locks[SHARD_COUNT];

    static inline size_t shard_of(size_t pos) noexcept {
        return pos & (SHARD_COUNT - 1);
    }

public:
    explicit MapaMasaSemantica(size_t initial_capacity = DEFAULT_MAX_POS)
        : _capacity(0)
    {
        allocate(initial_capacity);
    }

    void pre_allocate(size_t max_pos) {
        if (_capacity >= max_pos) return;
        allocate(max_pos);
    }

    size_t get_capacity() const noexcept { return _capacity; }

    void registrar_masa(int pos, float masa) {
        if (pos < 0 || _capacity == 0) return;
        size_t upos = static_cast<size_t>(pos);
        if (upos >= _capacity) return;

        size_t shard = shard_of(upos);
        NaiScopedLock guard(_locks[shard]);
        _masas[upos].store(masa, std::memory_order_relaxed);
    }

    [[nodiscard]] float obtener_masa(int pos) const noexcept {
        if (pos < 0 || _capacity == 0) return 1.0f;
        size_t upos = static_cast<size_t>(pos);
        if (upos >= _capacity) return 1.0f;
        return _masas[upos].load(std::memory_order_relaxed);
    }

    // [NAI] Desplaza las masas cuando el KV Cache mueve las posiciones
    // de los tokens (K-Shift). Usado por el Desalojo Gravitatorio.
    void shift_masa(int start_pos, int end_pos, int delta) {
        if (delta == 0 || _capacity == 0) return;
        if (start_pos < 0 || end_pos > (int)_capacity) return;

        if (delta > 0) {
            for (int i = end_pos - 1; i >= start_pos; --i) {
                if (i + delta < (int)_capacity) {
                    float m = _masas[i].load(std::memory_order_relaxed);
                    _masas[i + delta].store(m, std::memory_order_relaxed);
                }
            }
            for (int i = start_pos; i < start_pos + delta; ++i) {
                if (i >= 0 && i < (int)_capacity) {
                    _masas[i].store(1.0f, std::memory_order_relaxed);
                }
            }
        } else {
            int abs_delta = -delta;
            for (int i = start_pos; i < end_pos; ++i) {
                if (i - abs_delta >= 0) {
                    float m = _masas[i].load(std::memory_order_relaxed);
                    _masas[i - abs_delta].store(m, std::memory_order_relaxed);
                }
            }
            for (int i = end_pos - abs_delta; i < end_pos; ++i) {
                if (i >= 0 && i < (int)_capacity) {
                    _masas[i].store(1.0f, std::memory_order_relaxed);
                }
            }
        }
    }

    void limpiar() {
        if (_capacity == 0) return;
        for (size_t s = 0; s < SHARD_COUNT; ++s) _locks[s].lock();
        for (size_t i = 0; i < _capacity; ++i) {
            _masas[i].store(1.0f, std::memory_order_relaxed);
        }
        for (size_t s = SHARD_COUNT; s > 0; --s) _locks[s - 1].unlock();
    }

private:
    void allocate(size_t max_pos) {
        size_t cap = 1;
        while (cap < max_pos) cap <<= 1;

        auto new_masas = std::unique_ptr<std::atomic<float>[]>(
            new std::atomic<float>[cap]);
        for (size_t i = 0; i < cap; ++i) {
            new_masas[i].store(1.0f, std::memory_order_relaxed);
        }
        _masas = std::move(new_masas);
        _capacity = cap;
    }
};

inline InerciaLatenteResult evaluar_inercia_tensor(
    const float * data, size_t n_embd, float umbral = 3.2f)
{
    InerciaLatenteResult res;
    if (!data || n_embd == 0) {
        res.es_autopista = false;
        res.varianza = 0.0f;
        res.entropia = 1.0f;
        return res;
    }

    double sum = 0.0;
    double sq_sum = 0.0;
    for (size_t i = 0; i < n_embd; ++i) {
        double val = static_cast<double>(data[i]);
        sum += val;
        sq_sum += val * val;
    }

    double mean = sum / n_embd;
    double variance = (sq_sum / n_embd) - (mean * mean);
    if (variance < 0.0) variance = 0.0;

    res.varianza = static_cast<float>(variance);
    res.es_autopista = (res.varianza > umbral);
    res.entropia = static_cast<float>(1.0 / (1.0 + res.varianza));
    return res;
}

inline float calcular_factor_gravitatorio(
    int posicion_token, int posicion_actual, float masa_semantica)
{
    int dt = posicion_actual - posicion_token;
    if (dt <= 128) return 1.0f;

    float alpha = 0.0005f;
    float g_factor = 1.0f / (1.0f + alpha * static_cast<float>(dt)
                             * (2.2f / (masa_semantica + 0.1f)));
    return g_factor;
}

inline void aplicar_atenuacion_gravitatoria(
    float * k_data, float * v_data, size_t dim,
    int posicion_token, int posicion_actual, float masa_semantica)
{
    if (!k_data || !v_data || dim == 0) return;

    float g_factor = calcular_factor_gravitatorio(
        posicion_token, posicion_actual, masa_semantica);
    if (g_factor >= 0.999f) return;

    for (size_t i = 0; i < dim; ++i) {
        k_data[i] *= g_factor;
        v_data[i] *= g_factor;
    }
}

inline void aplicar_atenuacion_tensor_fp16(
    ggml_fp16_t * k_data, ggml_fp16_t * v_data, size_t dim, float g_factor)
{
    if (!k_data || !v_data || dim == 0 || g_factor >= 0.999f) return;
    for (size_t i = 0; i < dim; ++i) {
        float k_val = ggml_fp16_to_fp32(k_data[i]) * g_factor;
        float v_val = ggml_fp16_to_fp32(v_data[i]) * g_factor;
        k_data[i] = ggml_fp32_to_fp16(k_val);
        v_data[i] = ggml_fp32_to_fp16(v_val);
    }
}

inline MapaMasaSemantica g_nai_mapa_masa;

} // namespace nai

inline InerciaLatenteResult evaluar_inercia_tensor(
    const float * data, size_t n_embd, float umbral = 3.2f)
{
    return nai::evaluar_inercia_tensor(data, n_embd, umbral);
}

#endif