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

int main() {
    llama_backend_init();
    const char * model_path = "qwen2.5-0.5b-instruct-q4_k_m.gguf";

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) return 1;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 512;
    cparams.n_seq_max = 2;
    cparams.n_threads = 4;

    llama_context * ctx = llama_init_from_model(model, cparams);
    
    common_params_sampling sparams;
    sparams.temp = 0.7f;
    common_sampler * smpl = common_sampler_init(model, sparams);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::string prompt = "<|im_start|>system\nEres un asistente util.<|im_end|>\n<|im_start|>user\nEscribe un ensayo sobre la historia de la computacion y la inteligencia artificial.";
    for(int i=0; i<5; i++) prompt += " Agrega mas contexto sobre el tema.";
    prompt += "<|im_end|>\n<|im_start|>assistant\n";

    std::vector<llama_token> tokens(prompt.size() + 16);
    int n_tokens = llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), false, true);
    tokens.resize(n_tokens);

    printf("\n==================================================\n");
    printf("   NAI BENCHMARK (Prompt: %d tokens)\n", n_tokens);
    printf("==================================================\n");

    llama_batch batch0 = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch0.token[i] = tokens[i]; batch0.pos[i] = i; batch0.n_seq_id[i] = 1;
        batch0.seq_id[i][0] = 0; batch0.logits[i] = (i == n_tokens - 1);
    }
    batch0.n_tokens = n_tokens;

    auto t0 = std::chrono::high_resolution_clock::now();
    llama_decode(ctx, batch0);
    auto t1 = std::chrono::high_resolution_clock::now();
    
    printf("[1] Prefill Normal (Seq 0):  %.2f ms (%.2f tok/s)\n", 
           time_ms(t0, t1), n_tokens / (time_ms(t0, t1) / 1000.0));
    llama_batch_free(batch0);

    llama_batch batch1 = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch1.token[i] = tokens[i]; batch1.pos[i] = i; batch1.n_seq_id[i] = 1;
        batch1.seq_id[i][0] = 1; batch1.logits[i] = (i == n_tokens - 1);
    }
    batch1.n_tokens = n_tokens;

    auto t2 = std::chrono::high_resolution_clock::now();
    llama_decode(ctx, batch1);
    auto t3 = std::chrono::high_resolution_clock::now();
    
    printf("[2] Prefill NAI Sharing(Seq 1): %.2f ms (%.2f tok/s)\n", 
           time_ms(t2, t3), n_tokens / (time_ms(t2, t3) / 1000.0));
    llama_batch_free(batch1);

    printf("\n[3] Generando 50 tokens (Midiendo Autopista)...\n");
    llama_pos n_past = n_tokens;
    int n_gen = 50;
    
    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_gen; i++) {
        llama_token id = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, id, true);
        
        llama_batch b = llama_batch_init(1, 0, 1);
        b.token[0] = id; b.pos[0] = n_past; b.n_seq_id[0] = 1;
        b.seq_id[0][0] = 0; b.logits[0] = true; b.n_tokens = 1;
        n_past++;
        llama_decode(ctx, b);
        llama_batch_free(b);
    }
    auto t5 = std::chrono::high_resolution_clock::now();
    
    printf("[3] Generacion: %.2f ms (%.2f tok/s)\n", 
           time_ms(t4, t5), n_gen / (time_ms(t4, t5) / 1000.0));

    printf("\n==================================================\n");
    printf("   TELEMETRIA DE LA AUTOPISTA (Samplers)\n");
    printf("==================================================\n");
    common_sampler_nai_stats_dump();

    common_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}