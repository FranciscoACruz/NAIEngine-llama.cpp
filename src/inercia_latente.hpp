#ifndef NAI_INERCIA_LATENTE_HPP
#define NAI_INERCIA_LATENTE_HPP

// =============================================================================
// [NAI] Mapa de masa semántica por (secuencia, posición)
//
// Cada llama_context tiene su propio mapa (no hay estado global), así que
// varios contextos y varias secuencias no se mezclan.
//
// Definición de "masa" (heurística, debe validarse con métricas de calidad):
//   masa(seq, pos) = entropía de Shannon normalizada, en [0, 1], de la
//                    distribución que el modelo predijo en esa posición.
//   - Masa baja  -> el modelo estaba seguro: texto predecible, candidato a
//                   desalojo cuando el contexto se llena.
//   - Masa alta  -> "nodo de conflicto": el modelo dudaba, se conserva.
//   - Posiciones sin logits (la mayor parte del prompt) reciben
//     MASA_PROTEGIDA (1.0), es decir, se tratan como las más importantes.
//
// El contexto no es thread-safe (igual que el resto de llama_context), por lo
// que no se usan locks.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace nai {

constexpr float MASA_PROTEGIDA = 1.0f;

// Entropía de Shannon normalizada (H / ln(n)) de softmax(logits), en [0, 1].
// Devuelve MASA_PROTEGIDA si los datos no son válidos.
inline float entropia_normalizada(const float * logits, size_t n) {
    if (!logits || n < 2) return MASA_PROTEGIDA;

    float l_max = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < n; ++i) {
        if (logits[i] > l_max) l_max = logits[i];
    }
    if (!std::isfinite(l_max)) return MASA_PROTEGIDA;

    // H = ln(Z) - sum(p_i * (l_i - l_max)),  con Z = sum(exp(l_i - l_max))
    double z  = 0.0;
    double sl = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(logits[i]) - l_max;
        if (std::isnan(d)) return MASA_PROTEGIDA;
        const double e = std::exp(d);
        z  += e;
        sl += e * d;
    }
    if (!(z > 0.0)) return MASA_PROTEGIDA;

    const double h     = std::log(z) - sl / z;
    const double h_max = std::log(static_cast<double>(n));
    double r = h / h_max;
    if (r < 0.0) r = 0.0;
    if (r > 1.0) r = 1.0;
    return static_cast<float>(r);
}

class MapaMasaSemantica {
public:
    void init(uint32_t n_seq, uint32_t n_pos) {
        _n_pos = n_pos;
        _masas.assign(n_seq, std::vector<float>(n_pos, MASA_PROTEGIDA));
    }

    bool activo() const noexcept { return !_masas.empty(); }

    void liberar() {
        _masas.clear();
        _masas.shrink_to_fit();
        _n_pos = 0;
    }

    void registrar(int32_t seq, int32_t pos, float masa) {
        if (!valido(seq, pos)) return;
        _masas[seq][pos] = masa;
    }

    float obtener(int32_t seq, int32_t pos) const {
        if (!valido(seq, pos)) return MASA_PROTEGIDA;
        return _masas[seq][pos];
    }

    void limpiar(int32_t seq) {
        if (seq < 0) {
            for (auto & v : _masas) std::fill(v.begin(), v.end(), MASA_PROTEGIDA);
            return;
        }
        if (seq < static_cast<int32_t>(_masas.size())) {
            std::fill(_masas[seq].begin(), _masas[seq].end(), MASA_PROTEGIDA);
        }
    }

    // Elimina [p0, p0 + n) y recorre hacia atrás las posiciones [p0 + n, p_end).
    // Equivale a seq_rm(p0, p0+n) + seq_add(p0+n, p_end, -n) en el KV cache.
    void quitar_y_recorrer(int32_t seq, int32_t p0, int32_t n, int32_t p_end) {
        if (seq < 0 || seq >= static_cast<int32_t>(_masas.size()) || n <= 0 || p0 < 0) return;
        auto & v = _masas[seq];
        const int32_t lim = std::min<int32_t>(p_end, static_cast<int32_t>(v.size()));
        for (int32_t i = p0 + n; i < lim; ++i) {
            v[i - n] = v[i];
        }
        for (int32_t i = std::max(p0, lim - n); i < lim; ++i) {
            v[i] = MASA_PROTEGIDA;
        }
    }

    // Ventana [p, p + n) con la menor masa media dentro de [p_min, p_end).
    // O(p_end - p_min) con suma deslizante. Devuelve -1 si no cabe ninguna.
    int32_t buscar_ventana_minima(int32_t seq, int32_t p_min, int32_t p_end, int32_t n) const {
        if (seq < 0 || seq >= static_cast<int32_t>(_masas.size()) || n <= 0) return -1;
        p_min = std::max<int32_t>(p_min, 0);
        p_end = std::min<int32_t>(p_end, static_cast<int32_t>(_n_pos));
        if (p_end - p_min < n) return -1;

        const auto & v = _masas[seq];
        double suma = 0.0;
        for (int32_t i = p_min; i < p_min + n; ++i) suma += v[i];

        double  mejor   = suma;
        int32_t p_mejor = p_min;
        for (int32_t p = p_min + 1; p + n <= p_end; ++p) {
            suma += v[p + n - 1] - v[p - 1];
            if (suma < mejor) { mejor = suma; p_mejor = p; }
        }
        return p_mejor;
    }

private:
    bool valido(int32_t seq, int32_t pos) const noexcept {
        return seq >= 0 && pos >= 0 &&
               seq < static_cast<int32_t>(_masas.size()) &&
               static_cast<uint32_t>(pos) < _n_pos;
    }

    uint32_t _n_pos = 0;
    std::vector<std::vector<float>> _masas;
};

} // namespace nai

#endif
