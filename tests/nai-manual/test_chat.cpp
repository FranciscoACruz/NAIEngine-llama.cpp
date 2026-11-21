#include "llama.h"
#include "common/sampling.h"
#include "common/common.h"

#include <cstdio>
#include <vector>
#include <string>

void generate(llama_context* ctx, common_sampler* smpl, int n_predict, llama_pos& n_past, llama_seq_id seq_id) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

    for (int i = 0; i < n_predict; i++) {
        llama_token id = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, id, true);

        bool is_eog = llama_vocab_is_eog(vocab, id);

        if (!is_eog) {
            char buf[128];
            int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
            if (n > 0) {
                fwrite(buf, 1, n, stdout);
                fflush(stdout);
            }
        }

        // SIEMPRE guardamos el token en el KV Cache (incluso el EOS)
        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.token[0] = id;
        batch.pos[0] = n_past;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = seq_id;
        batch.logits[0] = true;
        batch.n_tokens = 1;

        n_past++;

        llama_decode(ctx, batch);
        llama_batch_free(batch);

        if (is_eog) {
            break; // Ahora sí, terminamos DESPUÉS de guardar el EOS
        }
    }
    printf("\n");
}

int main() {
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file("qwen2.5-0.5b-instruct-q4_k_m.gguf", mparams);
    if (!model) return 1;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 1024;
    cparams.n_batch = 512;
    cparams.n_seq_max = 1;
    cparams.n_threads = 4;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) return 1;

    common_params_sampling sparams;
    sparams.temp = 0.7f;
    common_sampler * smpl = common_sampler_init(model, sparams);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // =========================================================================
    // TURNO 1
    // =========================================================================
    printf("\n[Usuario]: Hola, mi nombre es Francisco. ¿Puedes saludarme?\n[Asistente]: ");
    std::string prompt1 = "<|im_start|>user\nHola, mi nombre es Francisco. ¿Puedes saludarme?<|im_end|>\n<|im_start|>assistant\n";
    
    std::vector<llama_token> tokens1(prompt1.size() + 16);
    int n_tokens1 = llama_tokenize(vocab, prompt1.data(), prompt1.size(), tokens1.data(), tokens1.size(), false, true);
    tokens1.resize(n_tokens1);

    llama_batch batch1 = llama_batch_init(n_tokens1, 0, 1);
    for (int i = 0; i < n_tokens1; i++) {
        batch1.token[i] = tokens1[i];
        batch1.pos[i] = i;
        batch1.n_seq_id[i] = 1;
        batch1.seq_id[i][0] = 0;
        batch1.logits[i] = (i == n_tokens1 - 1);
    }
    batch1.n_tokens = n_tokens1;

    llama_pos n_past = n_tokens1;
    llama_decode(ctx, batch1);
    llama_batch_free(batch1);

    generate(ctx, smpl, 60, n_past, 0);

    // =========================================================================
    // TURNO 2
    // =========================================================================
    printf("\n[Usuario]: ¿Te acuerdas cómo me llamo?\n[Asistente]: ");
    std::string prompt2 = "<|im_start|>user\n¿Te acuerdas cómo me llamo?<|im_end|>\n<|im_start|>assistant\n";
    
    std::vector<llama_token> tokens2(prompt2.size() + 16);
    int n_tokens2 = llama_tokenize(vocab, prompt2.data(), prompt2.size(), tokens2.data(), tokens2.size(), false, true);
    tokens2.resize(n_tokens2);

    llama_batch batch2 = llama_batch_init(n_tokens2, 0, 1);
    for (int i = 0; i < n_tokens2; i++) {
        batch2.token[i] = tokens2[i];
        batch2.pos[i] = n_past + i;
        batch2.n_seq_id[i] = 1;
        batch2.seq_id[i][0] = 0;
        batch2.logits[i] = (i == n_tokens2 - 1);
    }
    batch2.n_tokens = n_tokens2;

    n_past += n_tokens2;
    llama_decode(ctx, batch2);
    llama_batch_free(batch2);

    generate(ctx, smpl, 60, n_past, 0);

    common_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}