#include "llama.h"
#include "nai-telemetry-hub.hpp"

#include <cstdio>
#include <vector>
#include <string>
#include <thread>
#include <chrono>

int main() {
    // 1. Inicializar backend
    llama_backend_init();

    // 2. Cargar el modelo Qwen 0.5B (que ya tienes en la raíz)
    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(
        "qwen2.5-0.5b-instruct-q4_k_m.gguf", mparams);
    if (!model) {
        std::fprintf(stderr, "ERROR: no se pudo cargar el modelo\n");
        llama_backend_free();
        return 1;
    }

    // 3. Crear contexto pequeño (n_ctx=512 es suficiente para el test)
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_batch = 512;
    cparams.n_threads = 4;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "ERROR: no se pudo crear el contexto\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // 4. Tokenizar un prompt corto
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::string prompt = "Hola, mi nombre es";
    std::vector<llama_token> tokens(prompt.size() + 16);
    int n_tokens = llama_tokenize(vocab, prompt.data(), prompt.size(),
                                  tokens.data(), tokens.size(), false, true);
    if (n_tokens < 0) {
        std::fprintf(stderr, "ERROR: tokenización falló\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    tokens.resize(n_tokens);

    // 5. Llenar el llama_batch directamente (sin helpers de common)
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_tokens - 1);
    }
    batch.n_tokens = n_tokens;

    // 6. Registrar callback en el Hub (contará e imprimirá eventos)
    int eventos_recibidos = 0;
    nai::g_nai_telemetry_hub.set_callback(
        [&eventos_recibidos](const nai::NaiTelemetryEvent& ev) {
            std::printf("[TELEMETRIA] action=%u detail_1=%u detail_2=%u session=%llu\n",
                        ev.action, ev.detail_1, ev.detail_2,
                        (unsigned long long)ev.session_handle);
            eventos_recibidos++;
        });

    // 7. Decodificar → esto debe disparar NAI_KV_TRACE_UBATCH_APPLIED + emit()
    std::printf("Prompt tiene %d tokens. Decodificando...\n", n_tokens);

    
    int ret = llama_decode(ctx, batch);
    std::printf("llama_decode retornó: %d\n", ret);

    // 8. Esperar al hilo consumidor (polling interno de 50ms)
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // 9. Resumen
    std::printf("\n=== RESUMEN ===\n");
    std::printf("Eventos recibidos en el callback: %d\n", eventos_recibidos);

    // 10. Cleanup
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return (eventos_recibidos > 0) ? 0 : 2;
}