#ifndef NAI_INERCIA_LATENTE_HPP
#define NAI_INERCIA_LATENTE_HPP

// =============================================================================
// [NAI] Mapa de masa semántica por (secuencia, posición)
//
// Cada llama_context tiene su propio mapa (no hay estado global), así que
// varios contextos y varias secuencias no se mezclan.
//
// Definición de "masa" (heurística, debe validarse con métricas de calidad):
//   masa(seq, pos) = sorpresa normalizada del token que REALMENTE quedó en
//                    esa posición, en [0, 1]:
//                        masa = -ln p(x_pos | x_<pos) / ln(n_vocab)
//   - Masa baja  -> el token era predecible (relleno, repetición de algo que
//                   ya está en el contexto): candidato a desalojo.
//   - Masa alta  -> el token aportó información nueva: se conserva.
//   - Posiciones sin predicción disponible (la mayor parte del prompt, porque
//     el servidor solo pide logits del último token) reciben MASA_PROTEGIDA
//     (1.0), es decir, nunca se desalojan antes que el texto generado.
//
// Nota sobre posiciones: los logits calculados para el token en `pos` predicen
// el token de `pos + 1`. Por eso la masa de `pos + 1` solo se conoce cuando ese
// token aparece (en el mismo batch o en el siguiente decode). Mientras tanto la
// fila de logits se guarda como "predicción pendiente" por secuencia.
//
// Versiones anteriores guardaban la entropía de la predicción en `pos`, lo que
// medía la incertidumbre sobre el token SIGUIENTE y la asignaba a la posición
// equivocada. entropia_normalizada() se conserva para diagnóstico.
//
// El contexto no es thread-safe (igual que el resto de llama_context), por lo
// que no se usan locks.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstring>
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

// log(sum(exp(logits))) estable. Devuelve NaN si los datos no son válidos.
inline double log_sum_exp(const float * logits, size_t n) {
    if (!logits || n == 0) return std::numeric_limits<double>::quiet_NaN();
    float l_max = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(logits[i])) return std::numeric_limits<double>::quiet_NaN();
        if (logits[i] > l_max) l_max = logits[i];
    }
    if (!std::isfinite(l_max)) return std::numeric_limits<double>::quiet_NaN();
    double z = 0.0;
    for (size_t i = 0; i < n; ++i) {
        z += std::exp(static_cast<double>(logits[i]) - l_max);
    }
    return static_cast<double>(l_max) + std::log(z);
}

// exp(x) sin ramas para x <= 0: 2^(x·log2e) = 2^n · 2^f, con n entero (armado por bits) y
// f en [0, 1) por un polinomio minimax de grado 5. Error relativo < 1e-6 en [-16, 0].
// Sin ramas ni llamadas, el compilador lo vectoriza (NEON en Apple Silicon, AVX en x86).
inline float exp_rapido(float x) {
    float t = x * 1.4426950408889634f;
    t = t < -126.0f ? -126.0f : t;
    const float fl = std::floor(t);
    const float f  = t - fl;
    float p = 1.8775767e-3f;
    p = p * f + 8.9893397e-3f;
    p = p * f + 5.5826318e-2f;
    p = p * f + 2.4015361e-1f;
    p = p * f + 6.9315308e-1f;
    p = p * f + 9.9999994e-1f;
    const uint32_t bits = static_cast<uint32_t>(static_cast<int32_t>(fl) + 127) << 23;
    float escala;
    std::memcpy(&escala, &bits, sizeof(escala));
    return p * escala;
}

// log(sum(exp(logits))) aproximado con corte: los logits a más de NAI_LSE_CORTE nats bajo el
// máximo aportan < e^-16 = 1.1e-7 cada uno; con |V| = 152k el error en lse es < 0.017 nats y en la
// masa normalizada < 0.0015. Dos pasadas con 16 acumuladores independientes (vectorizables):
// máximo y suma de exp_rapido. Los NaN se ignoran; si todo es NaN o -inf devuelve NaN.
// Costo medido con |V| = 152k: ~0.15 ms (logits con pico) a ~0.46 ms (logits planos) por token.
constexpr float NAI_LSE_CORTE = 16.0f;

inline float log_sum_exp_rapido(const float * logits, size_t n) {
    if (!logits || n == 0) return std::numeric_limits<float>::quiet_NaN();
    constexpr int K = 16;
    float m[K];
    for (int k = 0; k < K; ++k) m[k] = -std::numeric_limits<float>::infinity();
    size_t i = 0;
    for (; i + K <= n; i += K) {
        for (int k = 0; k < K; ++k) m[k] = logits[i + k] > m[k] ? logits[i + k] : m[k];
    }
    float l_max = -std::numeric_limits<float>::infinity();
    for (int k = 0; k < K; ++k) l_max = m[k] > l_max ? m[k] : l_max;
    for (; i < n; ++i) l_max = logits[i] > l_max ? logits[i] : l_max;
    if (!std::isfinite(l_max)) return std::numeric_limits<float>::quiet_NaN();

    float z[K] = {0.0f};
    i = 0;
    for (; i + K <= n; i += K) {
        for (int k = 0; k < K; ++k) {
            const float x = logits[i + k] - l_max;
            z[k] += x > -NAI_LSE_CORTE ? exp_rapido(x) : 0.0f;
        }
    }
    float zt = 0.0f;
    for (int k = 0; k < K; ++k) zt += z[k];
    for (; i < n; ++i) {
        const float x = logits[i] - l_max;
        zt += x > -NAI_LSE_CORTE ? exp_rapido(x) : 0.0f;
    }
    return l_max + std::log(zt);
}

// Sorpresa normalizada -ln p(token) / ln(n), recortada a [0, 1], dado el lse ya calculado.
inline float sorpresa_desde_lse(float lse, float logit_token, size_t n) {
    if (n < 2 || !std::isfinite(lse) || logit_token != logit_token) return MASA_PROTEGIDA;
    if (logit_token == -std::numeric_limits<float>::infinity()) return MASA_PROTEGIDA;
    float r = (lse - logit_token) / std::log(static_cast<float>(n));
    return r < 0.0f ? 0.0f : (r > 1.0f ? 1.0f : r);
}

// Sorpresa normalizada -ln p(token) / ln(n), recortada a [0, 1].
// Un token tan probable como en una distribución uniforme (o menos) da 1.0.
// Devuelve MASA_PROTEGIDA si los datos no son válidos.
inline float sorpresa_normalizada(const float * logits, size_t n, int32_t token) {
    if (!logits || n < 2 || token < 0 || static_cast<size_t>(token) >= n) return MASA_PROTEGIDA;
    return sorpresa_desde_lse(log_sum_exp_rapido(logits, n), logits[token], n);
}

class MapaMasaSemantica {
public:
    void init(uint32_t n_seq, uint32_t n_pos) {
        _n_pos = n_pos;
        _masas.assign(n_seq, std::vector<float>(n_pos, MASA_PROTEGIDA));
        _pend.assign(n_seq, Pendiente{});
    }

    bool activo() const noexcept { return !_masas.empty(); }

    void liberar() {
        _masas.clear();
        _masas.shrink_to_fit();
        _pend.clear();
        _pend.shrink_to_fit();
        _n_pos = 0;
    }

    // Guarda la fila de logits que predice el token de `pos_siguiente` en `seq`.
    // Se resuelve cuando ese token aparezca (resolver_pendiente).
    void guardar_pendiente(int32_t seq, int32_t pos_siguiente, const float * logits, size_t n_vocab) {
        if (!valido(seq, pos_siguiente) || !logits || n_vocab < 2) {
            descartar_pendiente(seq);
            return;
        }
        auto & p = _pend[seq];
        p.pos = pos_siguiente;
        p.lse = log_sum_exp_rapido(logits, n_vocab);
        p.fila.assign(logits, logits + n_vocab);
    }

    // Si hay una predicción pendiente para (seq, pos), calcula la sorpresa de
    // `token` y la registra como masa de esa posición. Devuelve true si resolvió.
    bool resolver_pendiente(int32_t seq, int32_t pos, int32_t token) {
        if (seq < 0 || seq >= static_cast<int32_t>(_pend.size())) return false;
        auto & p = _pend[seq];
        if (p.pos < 0 || p.pos != pos) return false;
        const float masa = (token < 0 || static_cast<size_t>(token) >= p.fila.size())
            ? MASA_PROTEGIDA : sorpresa_desde_lse(p.lse, p.fila[token], p.fila.size());
        registrar(seq, pos, masa);
        p.pos = -1; // se conserva la memoria de la fila para reutilizarla
        return true;
    }

    void descartar_pendiente(int32_t seq) {
        if (seq >= 0 && seq < static_cast<int32_t>(_pend.size())) _pend[seq].pos = -1;
    }

    int32_t pos_pendiente(int32_t seq) const {
        if (seq < 0 || seq >= static_cast<int32_t>(_pend.size())) return -1;
        return _pend[seq].pos;
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
            for (auto & p : _pend) p.pos = -1;
            return;
        }
        if (seq < static_cast<int32_t>(_masas.size())) {
            std::fill(_masas[seq].begin(), _masas[seq].end(), MASA_PROTEGIDA);
            _pend[seq].pos = -1;
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
        // la predicción pendiente se recorre igual que el KV cache
        auto & p = _pend[seq];
        if (p.pos >= p0 + n) {
            p.pos -= n;
        } else if (p.pos >= p0) {
            p.pos = -1;
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

    struct Pendiente {
        int32_t            pos = -1; // posición cuyo token predice `fila` (-1 = ninguna)
        float              lse = 0.0f; // log-sum-exp de `fila`, calculado al guardar
        std::vector<float> fila;     // logits (n_vocab); memoria reutilizada entre pasos
    };

    uint32_t _n_pos = 0;
    std::vector<std::vector<float>> _masas;
    std::vector<Pendiente>          _pend;  // una predicción pendiente por secuencia
};

} // namespace nai

#endif
