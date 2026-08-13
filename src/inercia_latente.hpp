#ifndef INERCIA_LATENTE_HPP
#define INERCIA_LATENTE_HPP

#include <cmath>
#include <cstdio>
#include <cstddef>
#include <vector>
#include "ggml.h"

struct InerciaLatenteResult {
    bool es_autopista;
    float varianza;
    float entropia;
};

namespace nai {

/**
 * @brief Evalúa si el tensor de la capa intermedia califica para Early Exit.
 */
inline InerciaLatenteResult evaluar_inercia_tensor(const float * data, size_t n_embd, float umbral = 3.2f) {
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

    res.varianza = static_cast<float>(variance);
    res.es_autopista = (res.varianza > umbral);
    res.entropia = static_cast<float>(1.0 / (1.0 + res.varianza));

    return res;
}

/**
 * @brief Calcula el factor de decaimiento gravitatorio según distancia temporal y masa semántica.
 */
inline float calcular_factor_gravitatorio(int posicion_token, int posicion_actual, float masa_semantica) {
    int dt = posicion_actual - posicion_token;
    if (dt <= 128) return 1.0f; // Umbral de protección: tokens recientes no sufren atenuación

    float alpha = 0.0005f; // Constante de gravedad temporal
    float g_factor = 1.0f / (1.0f + alpha * static_cast<float>(dt) * (2.2f / (masa_semantica + 0.1f)));
    return g_factor;
}

/**
 * @brief Aplica una atenuación gravitatoria directa sobre búfers planos FP32 de K y V.
 */
inline void aplicar_atenuacion_gravitatoria(
    float * k_data, 
    float * v_data, 
    size_t dim, 
    int posicion_token, 
    int posicion_actual, 
    float masa_semantica) 
{
    if (!k_data || !v_data || dim == 0) return;

    float g_factor = calcular_factor_gravitatorio(posicion_token, posicion_actual, masa_semantica);
    if (g_factor >= 0.999f) return; // No gasta ciclos si la atenuación es despreciable

    for (size_t i = 0; i < dim; ++i) {
        k_data[i] *= g_factor;
        v_data[i] *= g_factor;
    }
}

/**
 * @brief Sobrecarga para aplicar atenuación sobre tensores GGML FP16 (Uso interno KV Cache de llama.cpp).
 */
inline void aplicar_atenuacion_tensor_fp16(
    ggml_fp16_t * k_data, 
    ggml_fp16_t * v_data, 
    size_t dim, 
    float g_factor) 
{
    if (!k_data || !v_data || dim == 0 || g_factor >= 0.999f) return;

    for (size_t i = 0; i < dim; ++i) {
        float k_val = ggml_fp16_to_fp32(k_data[i]) * g_factor;
        float v_val = ggml_fp16_to_fp32(v_data[i]) * g_factor;
        k_data[i] = ggml_fp32_to_fp16(k_val);
        v_data[i] = ggml_fp32_to_fp16(v_val);
    }
}

/**
 * @brief Administrador global ligero del Mapa de Masa Semántica NAI para el KV Cache.
 */
class MapaMasaSemantica {
public:
    std::vector<float> masas;

    void registrar_masa(int pos, float masa) {
        if (pos >= static_cast<int>(masas.size())) {
            masas.resize(pos + 1024, 1.0f); // Expansión dinámica en bloques
        }
        masas[pos] = masa;
    }

    float obtener_masa(int pos) const {
        if (pos < 0 || pos >= static_cast<int>(masas.size())) return 1.0f; // Masa neutra por defecto
        return masas[pos];
    }

    void limpiar() {
        masas.clear();
    }
};

/**
 * @brief Instancia global compartida para el registro de masa semántica NAI.
 * Se declara DESPUÉS de la definición de la clase MapaMasaSemantica.
 */
inline MapaMasaSemantica g_nai_mapa_masa;

} // namespace nai

// Mantener función global por compatibilidad
inline InerciaLatenteResult evaluar_inercia_tensor(const float * data, size_t n_embd, float umbral = 3.2f) {
    return nai::evaluar_inercia_tensor(data, n_embd, umbral);
}

#endif // INERCIA_LATENTE_HPP