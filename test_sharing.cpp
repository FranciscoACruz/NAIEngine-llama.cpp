#include "llama.h"
#include <cstdio>
#include <vector>
#include <string>

int main() {
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file("qwen2.5-0.5b-instruct-q4_k_m.gguf", mparams);
    if (!model) return 1;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_batch = 512;
    cparams.n_seq_max = 2;
    cparams.n_threads = 4;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) return 1;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::string prompt = "Hola, mi nombre es";
    std::vector<llama_token> tokens(prompt.size() + 16);
    int n_tokens = llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), false, true);
    tokens.resize(n_tokens);

    auto create_batch = [&](llama_seq_id seq_id) {
        llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        for (int i = 0; i < n_tokens; i++) {
            batch.token[i] = tokens[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = seq_id;
            batch.logits[i] = (i == n_tokens - 1);
        }
        batch.n_tokens = n_tokens;
        return batch;
    };

    printf("\n--- Decodificando seq 0 (Primera vez) ---\n");
    llama_batch batch0 = create_batch(0);
    llama_decode(ctx, batch0);
    llama_batch_free(batch0);

    printf("\n--- Decodificando seq 1 (Segunda vez, mismo prompt) ---\n");
    llama_batch batch1 = create_batch(1);
    llama_decode(ctx, batch1);
    llama_batch_free(batch1);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}