#pragma once

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
    // Palabras que indican lenguaje, entorno o restricción concreta (sin acentos, minúsculas).
    std::vector<std::string> anclajes_tecnicos = {
        // lenguajes
        "c", "c++", "c#", "python", "java", "javascript", "js", "typescript", "ts", "go", "golang",
        "rust", "php", "ruby", "kotlin", "swift", "dart", "bash", "shell", "powershell", "sql", "r",
        // datos y formatos
        "json", "yaml", "xml", "csv", "excel", "html", "css", "markdown",
        // plataformas y entorno
        "linux", "windows", "macos", "android", "ios", "docker", "kubernetes", "aws", "azure", "gcp",
        "react", "node", "django", "flask", ".net", "dotnet", "postgres", "mysql", "sqlserver",
        // conceptos técnicos
        "struct", "class", "clase", "funcion", "function", "error", "bug", "memoria", "algoritmo",
        "http", "api", "endpoint", "query", "consulta", "servidor", "server"
    };

    // Verbos de petición abierta en español e inglés (sin acentos).
    std::vector<std::string> verbos_abiertos = {
        "hazme", "haz", "crea", "mejora", "optimiza", "ayudame", "analiza", "como",
        "make", "create", "improve", "optimize", "help", "analyze", "how", "build"
    };

    // Minúsculas + supresión de diacríticos del español en UTF-8
    // (Á/á -> a, É/é -> e, Í/í -> i, Ó/ó -> o, Ú/ú/Ü/ü -> u, Ñ/ñ -> n).
    // El resto de bytes no ASCII se conserva sin cambios.
    static std::string a_minusculas(const std::string & str) {
        std::string res;
        res.reserve(str.size());
        for (size_t i = 0; i < str.size(); ++i) {
            const unsigned char c = static_cast<unsigned char>(str[i]);
            if (c == 0xC3 && i + 1 < str.size()) {
                const unsigned char d = static_cast<unsigned char>(str[i + 1]) | 0x20; // a minúscula
                char base = 0;
                switch (d) {
                    case 0xA1: base = 'a'; break; // á
                    case 0xA9: base = 'e'; break; // é
                    case 0xAD: base = 'i'; break; // í
                    case 0xB3: base = 'o'; break; // ó
                    case 0xBA: case 0xBC: base = 'u'; break; // ú ü
                    case 0xB1: base = 'n'; break; // ñ
                    default: break;
                }
                if (base) {
                    res.push_back(base);
                    ++i;
                    continue;
                }
            }
            res.push_back(c < 0x80 ? static_cast<char>(std::tolower(c)) : static_cast<char>(c));
        }
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

        // 2. Caso amplio (>= 7 palabras) sin entorno explícito: se responde de forma general
        //    y solo se avisa al usuario. NO se inyecta ningún supuesto en su mensaje
        //    (antes se añadía "Entorno: C++ ...", lo que cambiaba la petición sin avisar).
        if (num_palabras >= 7 && conteo_anclajes == 0 && tiene_verbo_abierto) {
            return {
                ModoTriaje::PROPUESTA_HIPOTESIS,
                "No detecté lenguaje, entorno ni restricción concretos; responderé de forma general. "
                "Si tienes alguno en mente, indícalo en tu siguiente mensaje.",
                ""
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