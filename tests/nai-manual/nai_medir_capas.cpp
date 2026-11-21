// [NAI] Medición de la autopista por capas ("conforme vas resolviendo, vas liberando").
//
// Para cada token generado lee el estado oculto a la salida de cada capa (l_out-N) y mide:
//   - Δh_l       = 1 - cos(h_l, h_{l-1})     cuánto cambia el estado entre capas ("inercia")
//   - sim_final  = cos(h_l, h_L)             qué tan cerca está ya del estado final
//   - lente      = argmax(E · norm(h_l))     qué token predeciría si saliera en la capa l
//                                            ("logit lens" con la cabeza de salida del modelo)
//   - capa de resolución = primera capa desde la que la lente coincide con la predicción
//                          final y ya no cambia: a partir de ahí las capas restantes
//                          no cambian la decisión y podrían liberarse.
//
// No modifica el motor: usa el callback de evaluación de ggml. Funciona con cualquier
// backend (CPU, Metal, CUDA, Vulkan) y con arquitecturas que nombren "l_out" y
// "result_norm" (llama, qwen2 y la mayoría).
//
// La lente es exacta en modelos con embeddings atados (tie_word_embeddings, p. ej.
// Qwen2.5 0.5B-3B). En modelos no atados se informa como aproximada.
//
// Uso:
//   nai_medir_capas -m modelo.gguf [-p "prompt"] [-n 64] [--lente-cada K]
//
// La lente cuesta n_vocab * n_embd operaciones por capa evaluada; --lente-cada K
// la evalúa cada K capas (y siempre en la última) para acelerar en modelos grandes.

#include "llama.h"
#include "../../src/llama-ext.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct captura {
    int n_layer = 0;
    int n_embd  = 0;
    std::vector<std::vector<float>> capas;  // [n_layer][n_embd] último token del ubatch
    std::vector<float>              final_norm;
    std::vector<bool>               vista;
};

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    auto * c = static_cast<captura *>(ud);
    const char * nombre = ggml_get_name(t);
    const bool es_capa  = strncmp(nombre, "l_out-", 6) == 0;
    const bool es_final = strcmp(nombre, "result_norm") == 0;
    if (ask) {
        return es_capa || es_final;
    }
    if (!(es_capa || es_final) || t->type != GGML_TYPE_F32 || t->ne[0] != c->n_embd) {
        return true;
    }
    const int64_t n_tok = t->ne[1];
    std::vector<float> fila(c->n_embd);
    const size_t off = (size_t) (n_tok - 1) * t->nb[1];
    if (t->buffer && !ggml_backend_buffer_is_host(t->buffer)) {
        ggml_backend_tensor_get(t, fila.data(), off, c->n_embd * sizeof(float));
    } else {
        memcpy(fila.data(), (const char *) t->data + off, c->n_embd * sizeof(float));
    }
    if (es_final) {
        c->final_norm = std::move(fila);
    } else {
        const int il = atoi(nombre + 6);
        if (il >= 0 && il < c->n_layer) {
            c->capas[il] = std::move(fila);
            c->vista[il] = true;
        }
    }
    return true;
}

static double coseno(const std::vector<float> & a, const std::vector<float> & b) {
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < a.size(); ++i) { ab += (double) a[i] * b[i]; aa += (double) a[i] * a[i]; bb += (double) b[i] * b[i]; }
    return (aa > 0 && bb > 0) ? ab / std::sqrt(aa * bb) : 0.0;
}

// RMSNorm sin peso
static std::vector<float> rms(const std::vector<float> & h, float eps = 1e-6f) {
    double s = 0;
    for (float v : h) s += (double) v * v;
    const float k = 1.0f / std::sqrt((float) (s / h.size()) + eps);
    std::vector<float> r(h.size());
    for (size_t i = 0; i < h.size(); ++i) r[i] = h[i] * k;
    return r;
}

static int argmax_lente(const std::vector<float> & E, int n_vocab, int n_embd, const std::vector<float> & x) {
    int best = 0;
    float best_v = -1e30f;
    for (int v = 0; v < n_vocab; ++v) {
        const float * e = E.data() + (size_t) v * n_embd;
        float acc = 0.0f;
        for (int i = 0; i < n_embd; ++i) acc += e[i] * x[i];
        if (acc > best_v) { best_v = acc; best = v; }
    }
    return best;
}

int main(int argc, char ** argv) {
    std::string ruta, prompt = "Explica en pocas palabras qué es una cuenta de ahorro:";
    int n_pred = 64, lente_cada = 1, n_threads = 4;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-m" && i + 1 < argc) ruta = argv[++i];
        else if (a == "-p" && i + 1 < argc) prompt = argv[++i];
        else if (a == "-n" && i + 1 < argc) n_pred = atoi(argv[++i]);
        else if (a == "-t" && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (a == "--lente-cada" && i + 1 < argc) lente_cada = std::max(1, atoi(argv[++i]));
    }
    if (ruta.empty()) {
        fprintf(stderr, "uso: %s -m modelo.gguf [-p prompt] [-n 64] [-t 4] [--lente-cada K]\n", argv[0]);
        return 1;
    }

    llama_backend_init();
    llama_log_set([](ggml_log_level lvl, const char * txt, void *) { if (lvl >= GGML_LOG_LEVEL_ERROR) fputs(txt, stderr); }, nullptr);

    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = [](float, void *) { return true; }; // sin puntos de progreso
    llama_model * model = llama_model_load_from_file(ruta.c_str(), mp);
    if (!model) { fprintf(stderr, "no se pudo cargar el modelo\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_embd  = llama_model_n_embd(model);
    const int n_layer = llama_model_n_layer(model);

    captura cap;
    cap.n_layer = n_layer;
    cap.n_embd  = n_embd;
    cap.capas.assign(n_layer, std::vector<float>(n_embd));
    cap.vista.assign(n_layer, false);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx             = 4096;
    cp.n_batch           = 512;
    cp.n_threads         = n_threads;
    cp.n_threads_batch   = n_threads;
    cp.cb_eval           = cb_eval;
    cp.cb_eval_user_data = &cap;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "no se pudo crear el contexto\n"); return 1; }

    // cabeza de salida (embeddings de entrada; exacta si el modelo los ata)
    std::vector<float> E((size_t) n_vocab * n_embd);
    if (llama_model_get_tok_embd(model, E.data()) == 0) { fprintf(stderr, "no se pudieron leer los embeddings\n"); return 1; }

    std::vector<llama_token> toks(prompt.size() + 16);
    const int n_p = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), toks.data(), (int) toks.size(), true, true);
    toks.resize(std::max(0, n_p));

    llama_batch b = llama_batch_get_one(toks.data(), (int) toks.size());
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "fallo al decodificar el prompt\n"); return 1; }

    std::vector<float> gamma;      // peso de la norma final, deducido de result_norm / rms(h_L)
    bool lente_exacta = false, lente_verificada = false;

    std::vector<int>    capa_res;  // capa de resolución por token
    std::vector<double> dh_sum(n_layer, 0.0), sim_sum(n_layer, 0.0);
    int n_medidos = 0;

    printf("modelo: %d capas, n_embd %d, vocab %d | prompt %d tokens | generando %d\n\n", n_layer, n_embd, n_vocab, (int) toks.size(), n_pred);
    printf("%-4s %-18s %-10s %s\n", "#", "token", "resuelto", "Δh por capa (·=<0.02 ▪=<0.10 █=≥0.10)");

    for (int it = 0; it < n_pred; ++it) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        int tok_final = 0;
        for (int v = 1; v < n_vocab; ++v) if (logits[v] > logits[tok_final]) tok_final = v;
        if (llama_vocab_is_eog(vocab, tok_final)) break;

        bool completo = !cap.final_norm.empty();
        for (bool v : cap.vista) completo = completo && v;
        if (completo) {
            if (gamma.empty()) {
                const auto r = rms(cap.capas[n_layer - 1]);
                gamma.resize(n_embd);
                for (int i = 0; i < n_embd; ++i) gamma[i] = std::fabs(r[i]) > 1e-6f ? cap.final_norm[i] / r[i] : 1.0f;
            }
            if (!lente_verificada) {
                lente_exacta = argmax_lente(E, n_vocab, n_embd, cap.final_norm) == tok_final;
                lente_verificada = true;
            }

            // lente por capa (de la última hacia atrás: buscamos desde dónde ya no cambia)
            int resuelto = n_layer - 1;
            for (int l = n_layer - 2; l >= 0; --l) {
                if (l % lente_cada != 0) continue;
                auto x = rms(cap.capas[l]);
                for (int i = 0; i < n_embd; ++i) x[i] *= gamma[i];
                if (argmax_lente(E, n_vocab, n_embd, x) != tok_final) break;
                resuelto = l;
            }
            capa_res.push_back(resuelto);

            std::string barra;
            for (int l = 0; l < n_layer; ++l) {
                const double sim_f = coseno(cap.capas[l], cap.capas[n_layer - 1]);
                const double dh    = l == 0 ? 0.0 : 1.0 - coseno(cap.capas[l], cap.capas[l - 1]);
                dh_sum[l] += dh; sim_sum[l] += sim_f;
                if (l > 0) barra += dh < 0.02 ? "·" : dh < 0.10 ? "▪" : "█";
            }
            n_medidos++;

            char pieza[64] = {0};
            const int np = llama_token_to_piece(vocab, tok_final, pieza, sizeof(pieza) - 1, 0, true);
            std::string txt(pieza, std::max(0, np));
            for (auto & ch : txt) if (ch == '\n') ch = ' ';
            if (txt.size() > 16) txt = txt.substr(0, 16);
            printf("%-4d %-18s capa %2d/%-2d %s\n", it, ("'" + txt + "'").c_str(), resuelto + 1, n_layer, barra.c_str());
        }

        std::fill(cap.vista.begin(), cap.vista.end(), false);
        cap.final_norm.clear();
        llama_token t = tok_final;
        if (llama_decode(ctx, llama_batch_get_one(&t, 1)) != 0) break;
    }

    if (n_medidos == 0) {
        printf("\nno se capturaron estados (¿la arquitectura usa otros nombres de tensores?)\n");
        return 1;
    }

    // resumen: distribución de capas de resolución ("niveles" de la autopista)
    printf("\nLente %s.\n", lente_exacta ? "EXACTA (embeddings atados)" : "APROXIMADA (modelo con cabeza de salida separada)");
    printf("\nResolución por cuartos de la red (qué fracción de tokens ya está decidida):\n");
    for (int q = 1; q <= 4; ++q) {
        const int lim = n_layer * q / 4;
        const int n = (int) std::count_if(capa_res.begin(), capa_res.end(), [&](int c) { return c + 1 <= lim; });
        printf("  hasta la capa %2d (%3d%%): %5.1f%% de los tokens\n", lim, 25 * q, 100.0 * n / capa_res.size());
    }
    double libre = 0;
    for (int c : capa_res) libre += (double) (n_layer - (c + 1)) / n_layer;
    printf("\nCapas que podrían liberarse en promedio (sin cambiar el token): %.1f%%\n", 100.0 * libre / capa_res.size());
    printf("(cota superior: no incluye el costo de decidir la salida ni el KV de las capas omitidas)\n");

    printf("\nPerfil medio por capa:\n  capa   Δh medio   sim. con estado final\n");
    for (int l = 0; l < n_layer; ++l) {
        printf("  %4d   %8.4f   %8.4f\n", l + 1, dh_sum[l] / n_medidos, sim_sum[l] / n_medidos);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
