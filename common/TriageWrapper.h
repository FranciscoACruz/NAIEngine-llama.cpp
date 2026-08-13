#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <cctype>

enum class ModoTriaje {
    DIRECTO_AUTOPISTA,  // Inferencia pesada inmediata
    INDAGACION_CORTA,   // Detener y pedir clarificación/deducción
    PROPUESTA_HIPOTESIS // Sugerir ruta deducida por defecto
};

struct ResultadoTriaje {
    ModoTriaje modo;
    std::string mensaje_respuesta; // Pregunta o hipótesis si no es directo
    std::string contexto_inyectado; // Anclaje semántico si se confirma
};

class TriageEngine {
private:
    std::vector<std::string> anclajes_tecnicos = {
        "c++", "python", "json", "sql", "struct", "class", 
        "función", "funcion", "error", "memoria", "algoritmo", "http", "api", "linux"
    };

    std::vector<std::string> verbos_abiertos = {
        "hazme", "crea", "mejora", "optimiza", "ayudame", "analiza", "como"
    };

    // Helper seguro para convertir UTF-8 / ASCII a minúsculas
    static std::string a_minusculas(const std::string & str) {
        std::string res = str;
        std::transform(res.begin(), res.end(), res.begin(), [](unsigned char c) {
            return std::tolower(c);
        });
        return res;
    }

    // Helper para limpiar signos de puntuación comunes alrededor de una palabra
    static std::string limpiar_token(const std::string & token) {
        size_t inicio = 0;
        size_t fin = token.size();

        while (inicio < fin && std::ispunct(static_cast<unsigned char>(token[inicio])) && token[inicio] != '+') {
            inicio++;
        }
        while (fin > inicio && std::ispunct(static_cast<unsigned char>(token[fin - 1])) && token[fin - 1] != '+') {
            fin--;
        }
        return token.substr(inicio, fin - inicio);
    }

public:
    ResultadoTriaje procesar_prompt(const std::string & prompt) {
        std::string texto = a_minusculas(prompt);

        // Tokenización real de palabras evitando falsos conteos por múltiples espacios
        std::stringstream ss(texto);
        std::string token;
        std::vector<std::string> palabras;

        while (ss >> token) {
            std::string token_limpio = limpiar_token(token);
            if (!token_limpio.empty()) {
                palabras.push_back(token_limpio);
            }
        }

        const size_t num_palabras = palabras.size();

        // Conteo exacto por palabra completa (Evita que "papaya" active "api")
        int conteo_anclajes = 0;
        bool tiene_verbo_abierto = false;

        for (const auto & p : palabras) {
            // Verificar anclajes técnicos
            for (const auto & kw : anclajes_tecnicos) {
                if (p == kw) {
                    conteo_anclajes++;
                    break;
                }
            }
            // Verificar verbos abiertos
            if (!tiene_verbo_abierto) {
                for (const auto & v : verbos_abiertos) {
                    if (p == v) {
                        tiene_verbo_abierto = true;
                        break;
                    }
                }
            }
        }

        // 1. Caso Ambiguo y Corto: Indagación Quirúrgica (< 7 palabras)
        if (num_palabras < 7 && tiene_verbo_abierto && conteo_anclajes == 0) {
            return {
                ModoTriaje::INDAGACION_CORTA,
                "Para darte la solución exacta a máxima velocidad y sin asumir de más: "
                "¿Podrías indicar el lenguaje, entorno o restricción principal de tu petición?",
                ""
            };
        }

        // 2. Caso Amplio pero Deduciable: Propuesta de Hipótesis Guiada (>= 7 palabras)
        if (num_palabras >= 7 && conteo_anclajes == 0 && tiene_verbo_abierto) {
            return {
                ModoTriaje::PROPUESTA_HIPOTESIS,
                "Dada tu consulta, deduzco que buscas una solución estándar optimizada en C++. "
                "¿Avanzamos por esa vía o prefieres acotar algún detalle?",
                "Entorno: C++ estándar, optimización de rendimiento y baja latencia. "
            };
        }

        // 3. Caso Específico: Pasar directo a la Autopista C++
        return {
            ModoTriaje::DIRECTO_AUTOPISTA,
            "",
            ""
        };
    }
};