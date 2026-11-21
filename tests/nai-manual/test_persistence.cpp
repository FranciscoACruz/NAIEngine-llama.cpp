#include "llama.h"
#include "common/sampling.h"
#include "common/common.h"

#include <cstdio>
#include <vector>
#include <string>
#include <sys/stat.h>   // Para el TTL de 24 horas
#include <ctime>

// Función auxiliar para generar texto
void generate(llama_context* ctx, common_sampler* smpl, int n_predict, llama_pos& n_past, llama_seq_id seq_id) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    for (int i = 0; i < n_predict; i++) {
        llama_token id = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, id, true);
        bool is_eog = llama_vocab_is_eog(vocab, id);
        if (!is_eog) {
            char buf[128];
            int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
            if (n > 0) { fwrite(buf, 1, n, stdout); fflush(stdout); }
        }
        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.token[0] = id; batch.pos[0] = n_past; batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = seq_id; batch.logits[0] = true; batch.n_tokens = 1;
        n_past++;
        llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (is_eog) break;
    }
    printf("\n");
}

int main() {
    llama_backend_init();
    const char * model_path = "qwen2.5-0.5b-instruct-q4_k_m.gguf";
    const char * state_path = "conversation_state.bin";
    const int TTL_24_HOURS = 86400; // 24 horas en segundos

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) return 1;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 1024;
    cparams.n_batch = 512;
    cparams.n_seq_max = 1;
    cparams.n_threads = 4;

    common_params_sampling sparams;
    sparams.temp = 0.7f;
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // =========================================================================
    // FASE 1: GUARDAR LA SESIÓN
    // =========================================================================
    printf("\n=== FASE 1: Creando y guardando la sesión ===\n");
    llama_context * ctx1 = llama_init_from_model(model, cparams);
    common_sampler * smpl1 = common_sampler_init(model, sparams);

    std::string prompt1 = "<|im_start|>user\nHola, mi nombre es Francisco.<|im_end|>\n<|im_start|>assistant\nHola Francisco, un gusto.<|im_end|>\n";
    std::vector<llama_token> tokens1(prompt1.size() + 16);
    int n_tokens1 = llama_tokenize(vocab, prompt1.data(), prompt1.size(), tokens1.data(), tokens1.size(), false, true);
    tokens1.resize(n_tokens1);

    llama_batch batch1 = llama_batch_init(n_tokens1, 0, 1);
    for (int i = 0; i < n_tokens1; i++) {
        batch1.token[i] = tokens1[i]; batch1.pos[i] = i; batch1.n_seq_id[i] = 1;
        batch1.seq_id[i][0] = 0; batch1.logits[i] = (i == n_tokens1 - 1);
    }
    batch1.n_tokens = n_tokens1;
    llama_decode(ctx1, batch1);
    llama_batch_free(batch1);

    // Guardar el estado (KV Cache + tokens) en disco
    size_t saved = llama_state_seq_save_file(ctx1, state_path, 0, tokens1.data(), n_tokens1);
    if (saved > 0) {
        printf("[Éxito] Estado guardado en %s (%zu bytes)\n", state_path, saved);
    } else {
        printf("[Error] No se pudo guardar el estado.\n");
        return 1;
    }

    common_sampler_free(smpl1);
    llama_free(ctx1);

    // =========================================================================
    // FASE 2: VERIFICAR TTL (24 HORAS)
    // =========================================================================
    printf("\n=== FASE 2: Verificando caducidad del archivo ===\n");
    struct stat file_stat;
    bool load_session = true;

    if (stat(state_path, &file_stat) == 0) {
        time_t now = time(nullptr);
        double age_seconds = difftime(now, file_stat.st_mtime);
        
        printf("[Info] El archivo tiene %.0f segundos de antigüedad.\n", age_seconds);
        
        // Lógica de 24 Horas
        if (age_seconds > TTL_24_HOURS) {
            printf("[TTL] El archivo tiene más de 24 horas. Borrando por seguridad...\n");
            std::remove(state_path);
            load_session = false;
        } else {
            printf("[TTL] El archivo es reciente. Procediendo a cargar...\n");
        }
    } else {
        printf("[Error] El archivo no existe.\n");
        load_session = false;
    }

    // =========================================================================
    // FASE 3: CARGAR LA SESIÓN EN UN MOTOR NUEVO
    // =========================================================================
    if (load_session) {
        printf("\n=== FASE 3: Cargando la sesión en un motor nuevo ===\n");
        llama_context * ctx2 = llama_init_from_model(model, cparams);
        common_sampler * smpl2 = common_sampler_init(model, sparams);

        // Cargar estado
        std::vector<llama_token> loaded_tokens(n_tokens1);
        size_t n_loaded = 0;
        llama_state_seq_load_file(ctx2, state_path, 0, loaded_tokens.data(), loaded_tokens.size(), &n_loaded);
        
        printf("[Éxito] Cargados %zu tokens. Preguntando al modelo...\n", n_loaded);
        
        // Preguntar el nombre basándonos en el contexto cargado
        llama_pos n_past = n_loaded;
        std::string prompt2 = "<|im_start|>user\n¿Te acuerdas cómo me llamo?<|im_end|>\n<|im_start|>assistant\n";
        std::vector<llama_token> tokens2(prompt2.size() + 16);
        int n_tokens2 = llama_tokenize(vocab, prompt2.data(), prompt2.size(), tokens2.data(), tokens2.size(), false, true);
        tokens2.resize(n_tokens2);

        llama_batch batch2 = llama_batch_init(n_tokens2, 0, 1);
        for (int i = 0; i < n_tokens2; i++) {
            batch2.token[i] = tokens2[i]; batch2.pos[i] = n_past + i; batch2.n_seq_id[i] = 1;
            batch2.seq_id[i][0] = 0; batch2.logits[i] = (i == n_tokens2 - 1);
        }
        batch2.n_tokens = n_tokens2;
        n_past += n_tokens2;
        llama_decode(ctx2, batch2);
        llama_batch_free(batch2);

        printf("[Asistente]: ");
        generate(ctx2, smpl2, 40, n_past, 0);

        common_sampler_free(smpl2);
        llama_free(ctx2);
    }

    llama_model_free(model);
    llama_backend_free();
    return 0;
}