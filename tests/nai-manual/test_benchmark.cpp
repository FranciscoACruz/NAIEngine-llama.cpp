#include "llama.h"
#include "common/sampling.h"
#include "common/common.h"
#include <cstdio>
#include <vector>
#include <string>
#include <chrono>

double time_ms(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void run_stress_test(llama_model * model, const llama_vocab * vocab, int n_seq_max, int tokens_per_user) {
    int n_ctx = n_seq_max * tokens_per_user;
    
    printf("\n--------------------------------------------------\n");
    printf("STRESS TEST: %d usuarios simultaneos (Contexto: %d tokens)\n", n_seq_max, n_ctx);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = 128; // Limitar el batch para no saturar la GPU Metal
    cparams.n_seq_max = n_seq_max;
    cparams.n_threads = 8;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        printf(" ❌ FALLO CRITICO: Memoria insuficiente (OOM) para esta configuracion.\n");
        return;
    }

    std::string prompt = "<|im_start|>user\nHola<|im_end|>\n<|im_start|>assistant\n";
    std::vector<llama_token> tokens(prompt.size() + 16);
    int n_tokens = llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), false, true);
    tokens.resize(n_tokens);

    // Prefill concurrente
    llama_batch batch = llama_batch_init(n_tokens * n_seq_max, 0, n_seq_max);
    int idx = 0;
    for (int u = 0; u < n_seq_max; u++) {
        for (int i = 0; i < n_tokens; i++) {
            batch.token[idx] = tokens[i]; batch.pos[idx] = i; 
            batch.n_seq_id[idx] = 1; batch.seq_id[idx][0] = u; 
            batch.logits[idx] = (i == n_tokens - 1); idx++;
        }
    }
    batch.n_tokens = idx;
    llama_decode(ctx, batch);
    llama_batch_free(batch);

    // Generación concurrente (10 pasos)
    int steps = 10;
    llama_batch gen_batch = llama_batch_init(n_seq_max, 0, n_seq_max);
    for (int u = 0; u < n_seq_max; u++) {
        gen_batch.token[u] = tokens[0]; gen_batch.pos[u] = n_tokens; 
        gen_batch.n_seq_id[u] = 1; gen_batch.seq_id[u][0] = u; gen_batch.logits[u] = true;
    }
    gen_batch.n_tokens = n_seq_max;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < steps; s++) {
        llama_decode(ctx, gen_batch);
        for(int u=0; u<n_seq_max; u++) gen_batch.pos[u]++;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    
    double ms = time_ms(t0, t1);
    double total_tokens = steps * n_seq_max;
    double tok_s_total = total_tokens / (ms / 1000.0);
    
    printf(" ✅ Throughput Total:    %7.2f tokens/segundo\n", tok_s_total);
    printf(" -  Velocidad por User:  %7.2f tokens/segundo\n", tok_s_total / n_seq_max);

    llama_batch_free(gen_batch);
    llama_free(ctx);
}

int main() {
    llama_backend_init();
    const char * model_path = "qwen2.5-0.5b-instruct-q4_k_m.gguf";

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) return 1;

    const llama_vocab * vocab = llama_model_get_vocab(model);

    printf("\n==================================================\n");
    printf("   PARTE 1: CARACTERISTICAS NAI (KV SHARING + AUTOPISTA)\n");
    printf("==================================================\n");

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 512;
    cparams.n_seq_max = 2; 
    cparams.n_threads = 4;
    llama_context * ctx = llama_init_from_model(model, cparams);
    
    common_params_sampling sparams;
    sparams.temp = 0.8f;
    common_sampler * smpl = common_sampler_init(model, sparams);

    std::string prompt = "<|im_start|>system\nEres un asistente util.<|im_end|>\n<|im_start|>user\nEscribe un ensayo sobre la historia de la computacion y la inteligencia artificial.<|im_end|>\n<|im_start|>assistant\n";
    std::vector<llama_token> tokens(prompt.size() + 16);
    int n_tokens = llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), false, true);
    tokens.resize(n_tokens);

    // 1. Prefill Normal (Procesa los 79 tokens)
    llama_batch batch0 = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch0.token[i] = tokens[i]; batch0.pos[i] = i; batch0.n_seq_id[i] = 1;
        batch0.seq_id[i][0] = 0; batch0.logits[i] = (i == n_tokens - 1);
    }
    batch0.n_tokens = n_tokens;
    auto t0 = std::chrono::high_resolution_clock::now();
    llama_decode(ctx, batch0);
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("[1] Prefill Normal (Seq 0):  %.2f ms (%.2f tok/s)\n", time_ms(t0, t1), n_tokens / (time_ms(t0, t1) / 1000.0));
    llama_batch_free(batch0);

    // 2. Prefill NAI Sharing (El KV se copia, SOLO procesamos el ultimo token)
    llama_batch batch1 = llama_batch_init(1, 0, 1);
    batch1.token[0] = tokens[n_tokens - 1]; 
    batch1.pos[0] = n_tokens - 1; 
    batch1.n_seq_id[0] = 1;
    batch1.seq_id[0][0] = 1; // Secuencia 1
    batch1.logits[0] = true;
    batch1.n_tokens = 1; // Solo 1 token!
    
    auto t2 = std::chrono::high_resolution_clock::now();
    llama_decode(ctx, batch1);
    auto t3 = std::chrono::high_resolution_clock::now();
    printf("[2] Prefill NAI Sharing(Seq 1): %.2f ms (El KV se copio y solo se proceso 1 token)\n", time_ms(t2, t3));
    llama_batch_free(batch1);

    // 3. Generación de Texto
    printf("\n--- GENERACION MASIVA (200 tokens) ---\n");
    printf("[Asistente]: ");
    fflush(stdout);
    llama_pos n_past = n_tokens;
    int n_gen = 200;
    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_gen; i++) {
        llama_token id = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, id, true);
        if (llama_vocab_is_eog(vocab, id)) break;
        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n > 0) { fwrite(buf, 1, n, stdout); fflush(stdout); }
        llama_batch b = llama_batch_init(1, 0, 1);
        b.token[0] = id; b.pos[0] = n_past; b.n_seq_id[0] = 1;
        b.seq_id[0][0] = 0; b.logits[0] = true; b.n_tokens = 1;
        n_past++;
        llama_decode(ctx, b);
        llama_batch_free(b);
    }
    auto t5 = std::chrono::high_resolution_clock::now();
    printf("\n\n[3] Velocidad Generacion: %.2f tok/s\n", n_gen / (time_ms(t4, t5) / 1000.0));

    // 4. Telemetría
    printf("\n--- TELEMETRIA AUTOPISTA ---\n");
    common_sampler_nai_stats_dump();
    fflush(stdout);

    common_sampler_free(smpl);
    llama_free(ctx);

    printf("\n==================================================\n");
    printf("   PARTE 2: TEST DE LIMITES DE CONCURRENCIA\n");
    printf("==================================================\n");
    
    run_stress_test(model, vocab, 1, 512);
    run_stress_test(model, vocab, 8, 512);
    run_stress_test(model, vocab, 32, 512);
    run_stress_test(model, vocab, 64, 512); // Reducimos contexto por usuario para que quepa
    run_stress_test(model, vocab, 128, 512); // 128 usuarios con contexto corto

    printf("\n==================================================\n");
    printf("   FIN DEL BENCHMARK COMPLETO\n");
    printf("==================================================\n");

    llama_model_free(model);
    llama_backend_free();
    return 0;
}