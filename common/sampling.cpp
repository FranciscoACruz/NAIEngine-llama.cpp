#include "sampling.h"
#include "common.h"
#include "fit.h"
#include "log.h"
#include "reasoning-budget.h"
#include "ggml.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>
#include <regex>
#include <unordered_map>
#include <vector>

// =================================================================
// 1. ESTRUCTURA NUMÉRICA Y EVALUADOR DE INERCIA (100% Determinista)
// =================================================================
struct DecisionInercia {
    bool    es_autopista; // true (1): Vía rápida | false (0): Requiere Samplers
    uint8_t codigo_razon; // Código numérico fijo: 1, 2, 3, 4 o 0
    float   margen;       // Métrica física calculada
};

static DecisionInercia evaluar_inercia_y_equivalencia(
    const llama_token_data_array & cur_p, 
    float epsilon_p = 0.02f,
    float delta_logit = 2.2f
) {
    if (cur_p.size <= 1) return { true, 1, 1.0f };

    float l1 = cur_p.data[0].logit, l2 = cur_p.data[1].logit;
    float margen_logits = l1 - l2;
    if (margen_logits > delta_logit) return { true, 2, margen_logits };

    float p1 = cur_p.data[0].p, p2 = cur_p.data[1].p;
    float margen_p = p1 - p2;
    if (p1 > 0.0f && std::abs(margen_p) < epsilon_p) return { true, 3, margen_p };

    if (cur_p.size >= 5 && p1 > 0.0f) {
        float suma_cola = 0.0f;
        for (size_t i = 1; i < 5; ++i) suma_cola += cur_p.data[i].p;
        if (p1 > 1.3f * suma_cola) return { true, 4, p1 - suma_cola };
    }

    return { false, 0, margen_p };
}

// Buffer circular optimizado para tokens históricos
template<typename T>
struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() { if (!sz) throw std::runtime_error("ring buffer empty"); return data[first]; }
    const T & front() const { if (!sz) throw std::runtime_error("ring buffer empty"); return data[first]; }
    T & back() { if (!sz) throw std::runtime_error("ring buffer empty"); return data[(first + sz - 1) % capacity]; }
    const T & back() const { if (!sz) throw std::runtime_error("ring buffer empty"); return data[(first + sz - 1) % capacity]; }

    void push_back(const T & value) {
        if (sz == capacity) first = (first + 1) % capacity;
        else sz++;
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (!sz) throw std::runtime_error("ring buffer empty");
        T val = data[first];
        first = (first + 1) % capacity;
        sz--;
        return val;
    }

    const T & rat(size_t i) const {
        if (i >= sz) throw std::runtime_error("ring buffer out of bounds");
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> res; res.reserve(sz);
        for (size_t i = 0; i < sz; i++) res.push_back(data[(first + i) % capacity]);
        return res;
    }

    void clear() { sz = 0; first = 0; pos = 0; }
    bool empty() const { return sz == 0; }
    size_t size() const { return sz; }

    size_t capacity = 0, sz = 0, first = 0, pos = 0;
    std::vector<T> data;
};

struct common_sampler {
    common_params_sampling params;
    struct llama_sampler * grmr;
    struct llama_sampler * rbudget;
    struct llama_sampler * chain;

    ring_buffer<llama_token> prev;
    std::vector<llama_token_data> cur;
    llama_token_data_array cur_p;

    void reset() {
        prev.clear();
        llama_sampler_reset(chain);
    }

    void set_logits(struct llama_context * ctx, int idx) {
        const float *       sampled_probs  = llama_get_sampled_probs_ith(ctx, idx);
        const float *       sampled_logits = llama_get_sampled_logits_ith(ctx, idx);
        const llama_token * sampled_ids    = llama_get_sampled_candidates_ith(ctx, idx);

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int n_vocab = llama_vocab_n_tokens(vocab);

        if (sampled_probs) {
            const uint32_t count = llama_get_sampled_probs_count_ith(ctx, idx);
            cur.resize(count);
            for (uint32_t i = 0; i < count; ++i) cur[i] = {sampled_ids[i], sampled_logits[i], sampled_probs[i]};
        } else if (sampled_logits) {
            const uint32_t count = llama_get_sampled_logits_count_ith(ctx, idx);
            cur.resize(count);
            for (uint32_t i = 0; i < count; i++) cur[i] = {sampled_ids[i], sampled_logits[i], 0.0f};
        } else {
            const auto * logits = llama_get_logits_ith(ctx, idx);
            GGML_ASSERT(logits != nullptr);
            cur.resize(n_vocab);
            for (llama_token id = 0; id < n_vocab; id++) cur[id] = {id, logits[id], 0.0f};
        }
        cur_p = { cur.data(), cur.size(), -1, false };
    }

    common_time_meas tm() { return common_time_meas(t_total_us, params.no_perf); }
    mutable int64_t t_total_us = 0;
};
// =================================================================
// 2. MUESTREO CON BYPASS DE AUTOPISTA (OPTIMIZACIÓN DE INERCIA)
// =================================================================
llama_token common_sampler_sample(struct common_sampler * gsm, struct llama_context * ctx, int idx, bool grammar_first) {
    const auto tm = gsm->tm();
    
    if (!gsm->params.backend_sampling) {
        gsm->set_logits(ctx, idx);
    }

    if (gsm->params.backend_sampling) {
        GGML_ASSERT(gsm->grmr == nullptr && "backend sampling is not supported with grammar");
        GGML_ASSERT(gsm->rbudget == nullptr && "backend sampling is not supported with reasoning budget");

        const auto * logits = llama_get_logits_ith(ctx, idx);
        GGML_ASSERT(logits != nullptr);

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int n_vocab = llama_vocab_n_tokens(vocab);

        gsm->cur.resize(n_vocab);
        for (llama_token id = 0; id < n_vocab; id++) {
            gsm->cur[id] = { id, logits[id], 0.0f };
        }
        gsm->cur_p = { gsm->cur.data(), gsm->cur.size(), -1, false };

        llama_sampler_apply(gsm->chain, &gsm->cur_p);
        GGML_ASSERT(gsm->cur_p.selected != -1);
        return gsm->cur_p.data[gsm->cur_p.selected].id;
    }

    // A. Aplicación de Gramática previa si grammar_first es true
    if (gsm->grmr && grammar_first) {
        llama_sampler_apply(gsm->grmr, &gsm->cur_p);
    }

    // B. Filtro de Presupuesto de Razonamiento (Reasoning Budget)
    if (gsm->rbudget) {
        llama_sampler_apply(gsm->rbudget, &gsm->cur_p);
    }

    // C. EVALUADOR ELÁSTICO DE INERCIA Y AUTOPISTA
    std::sort(gsm->cur_p.data, gsm->cur_p.data + gsm->cur_p.size, [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit > b.logit;
    });

    DecisionInercia dec = evaluar_inercia_y_equivalencia(gsm->cur_p);

    if (dec.es_autopista) {
        LOG_DBG("%s: [Inercia Autopista] Codigo: %d, Margen: %.4f -> Token direct: %d\n",
                __func__, dec.codigo_razon, dec.margen, gsm->cur_p.data[0].id);
        return gsm->cur_p.data[0].id;
    }

    // D. Aplicación de Gramática si no fue aplicada primero
    if (gsm->grmr && !grammar_first) {
        llama_sampler_apply(gsm->grmr, &gsm->cur_p);
    }

    // E. Cadena estándar de Samplers pesados
    llama_sampler_apply(gsm->chain, &gsm->cur_p);

    GGML_ASSERT(gsm->cur_p.selected != -1);

    const llama_token id = gsm->cur_p.data[gsm->cur_p.selected].id;

    return id;
}
std::string common_params_sampling::print() const {
    char result[1024];
    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
            "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = %.3f, top_n_sigma = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f, adaptive_target = %.3f, adaptive_decay = %.3f",
            penalty_last_n, penalty_repeat, penalty_freq, penalty_present,
            dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
            top_k, top_p, min_p, xtc_probability, xtc_threshold, typ_p, top_n_sigma, temp,
            mirostat, mirostat_eta, mirostat_tau, adaptive_target, adaptive_decay);
    return std::string(result);
}

struct common_sampler * common_sampler_init(
        const struct llama_model * model,
        struct common_params_sampling & params) {
    if (!std::isfinite(params.penalty_repeat) || params.penalty_repeat <= 0.0f || !std::isfinite(1.0f/params.penalty_repeat)) {
        throw std::invalid_argument("penalty_repeat must be finite and greater than 0");
    }
    if (!std::isfinite(params.penalty_freq)) throw std::invalid_argument("penalty_freq must be finite");
    if (!std::isfinite(params.penalty_present)) throw std::invalid_argument("penalty_present must be finite");

    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();
    lparams.no_perf = params.no_perf;

    llama_sampler * grmr = nullptr;
    llama_sampler * rbudget = nullptr;
    llama_sampler * chain = llama_sampler_chain_init(lparams);

    std::vector<llama_sampler *> samplers;
    const std::string & grammar_str = common_grammar_value(params.grammar);

    if (grammar_str.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
        grmr = llama_sampler_init_llg(vocab, "lark", grammar_str.c_str());
#else
        GGML_ABORT("llguidance (cmake -DLLAMA_LLGUIDANCE=ON) is not enabled");
#endif
    } else {
        std::vector<std::string> trigger_patterns;
        std::vector<llama_token> trigger_tokens;
        for (const auto & trigger : params.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                    trigger_patterns.push_back(regex_escape(trigger.value));
                    break;
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                    trigger_patterns.push_back(trigger.value);
                    break;
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL: {
                    const auto & pattern = trigger.value;
                    std::string anchored = "^$";
                    if (!pattern.empty()) {
                        anchored = (pattern.front() != '^' ? "^" : "") + pattern + (pattern.back() != '$' ? "$" : "");
                    }
                    trigger_patterns.push_back(anchored);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                    trigger_tokens.push_back(trigger.token);
                    break;
                default:
                    GGML_ASSERT(false && "unknown trigger type");
            }
        }

        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & regex : trigger_patterns) {
            trigger_patterns_c.push_back(regex.c_str());
        }

        if (!grammar_str.empty()) {
            if (params.grammar_lazy) {
                grmr = llama_sampler_init_grammar_lazy_patterns(vocab, grammar_str.c_str(), "root",
                        trigger_patterns_c.data(), trigger_patterns_c.size(),
                        trigger_tokens.data(), trigger_tokens.size());
            } else {
                grmr = llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root");
            }
        }
    }

    if (!grmr && !grammar_str.empty()) {
        throw std::runtime_error("failed to parse grammar");
    }

    std::vector<llama_token> prefill_tokens;
    if (!params.generation_prompt.empty()) {
        GGML_ASSERT(vocab != nullptr);
        auto tokens = common_tokenize(vocab, params.generation_prompt, false, true);
        for (size_t i = 0; i < tokens.size(); i++) {
            std::string piece = common_token_to_piece(vocab, tokens[i], true);
            if (i == 0 && std::isspace(piece[0]) && !std::isspace(params.generation_prompt[0])) continue;
            LOG_DBG("%s: prefill token: %d = %s\n", __func__, tokens[i], piece.c_str());
            prefill_tokens.push_back(tokens[i]);
        }
    }

    if (grmr && !params.grammar_lazy && common_grammar_needs_prefill(params.grammar)) {
        try {
            for (const auto & token : prefill_tokens) {
                llama_sampler_accept(grmr, token);
                LOG_DBG("%s: grammar accepted prefill token (%d)\n", __func__, token);
            }
        } catch (std::exception &e) {
            LOG_ERR("%s: error initializing grammar sampler for grammar:\n%s\n\nGeneration prompt:\n'%s'\n", __func__,
                common_grammar_value(params.grammar).c_str(), params.generation_prompt.c_str());
            throw e;
        }
    }

    if (!params.reasoning_budget_start.empty() && !params.reasoning_budget_end.empty() && 
        (params.grammar_lazy || params.reasoning_budget_tokens >= 0 || params.reasoning_control)) {
        rbudget = common_reasoning_budget_init(
            vocab,
            {params.reasoning_budget_start},
            params.reasoning_budget_end,
            params.reasoning_budget_forced,
            params.reasoning_budget_tokens < 0 ? INT_MAX : params.reasoning_budget_tokens);

        for (const auto & token : prefill_tokens) {
            llama_sampler_accept(rbudget, token);
            LOG_DBG("%s: reasoning-budget accepted prefill token (%d)\n", __func__, token);
        }
    }

    {
        std::vector<llama_logit_bias> merged = params.logit_bias;
        int32_t n_suppress = 0;
        const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
        for (int32_t i = 0; i < n_suppress; ++i) merged.push_back({ suppress[i], -INFINITY });
        if (!merged.empty()) {
            samplers.push_back(llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), merged.size(), merged.data()));
        }
    }

    if (params.mirostat == 0) {
        bool use_adaptive_p = false;
        for (const auto & cnstr : params.samplers) {
            switch (cnstr) {
                case COMMON_SAMPLER_TYPE_DRY: {
                    std::vector<const char *> c_breakers;
                    c_breakers.reserve(params.dry_sequence_breakers.size());
                    for (const auto & str : params.dry_sequence_breakers) c_breakers.push_back(str.c_str());
                    samplers.push_back(llama_sampler_init_dry(vocab, params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()));
                    break;
                }
                case COMMON_SAMPLER_TYPE_TOP_K:       samplers.push_back(llama_sampler_init_top_k(params.top_k)); break;
                case COMMON_SAMPLER_TYPE_TOP_P:       samplers.push_back(llama_sampler_init_top_p(params.top_p, params.min_keep)); break;
                case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: samplers.push_back(llama_sampler_init_top_n_sigma(params.top_n_sigma)); break;
                case COMMON_SAMPLER_TYPE_MIN_P:       samplers.push_back(llama_sampler_init_min_p(params.min_p, params.min_keep)); break;
                case COMMON_SAMPLER_TYPE_XTC:         samplers.push_back(llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold, params.min_keep, params.seed)); break;
                case COMMON_SAMPLER_TYPE_TYPICAL_P:   samplers.push_back(llama_sampler_init_typical(params.typ_p, params.min_keep)); break;
                case COMMON_SAMPLER_TYPE_TEMPERATURE: samplers.push_back(llama_sampler_init_temp_ext(params.temp, params.dynatemp_range, params.dynatemp_exponent)); break;
                case COMMON_SAMPLER_TYPE_INFILL:      samplers.push_back(llama_sampler_init_infill(vocab)); break;
                case COMMON_SAMPLER_TYPE_PENALTIES:   samplers.push_back(llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present)); break;
                case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  use_adaptive_p = true; break;
                default: GGML_ASSERT(false && "unknown sampler type");
            }
        }
        if (use_adaptive_p) {
            samplers.push_back(llama_sampler_init_adaptive_p(params.adaptive_target, params.adaptive_decay, params.seed));
        } else {
            samplers.push_back(llama_sampler_init_dist(params.seed));
        }
    } else if (params.mirostat == 1) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), params.seed, params.mirostat_tau, params.mirostat_eta, 100));
    } else if (params.mirostat == 2) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
    } else {
        GGML_ASSERT(false && "unknown mirostat version");
    }

    for (auto * smpl : samplers) llama_sampler_chain_add(chain, smpl);

    if (grmr && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with grammar, disabling\n", __func__);
        params.backend_sampling = false;
    }
    if (rbudget && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with reasoning budget, disabling\n", __func__);
        params.backend_sampling = false;
    }

    return new common_sampler { params, grmr, rbudget, chain, ring_buffer<llama_token>(std::max(32, params.n_prev)), {}, {} };
}

void common_sampler_free(struct common_sampler * gsm) {
    if (!gsm) return;
    if (gsm->grmr)    llama_sampler_free(gsm->grmr);
    if (gsm->rbudget) llama_sampler_free(gsm->rbudget);
    if (gsm->chain)   llama_sampler_free(gsm->chain);
    delete gsm;
}

void common_sampler_accept(struct common_sampler * gsm, llama_token token, bool accept_hardware) {
    const auto tm = gsm->tm();
    if (accept_hardware) {
        if (gsm->grmr)    llama_sampler_accept(gsm->grmr, token);
        if (gsm->rbudget) llama_sampler_accept(gsm->rbudget, token);
    }
    llama_sampler_accept(gsm->chain, token);
    gsm->prev.push_back(token);
}

void common_sampler_sample_and_accept(struct common_sampler * gsm, struct llama_context * ctx, int idx) {
    const llama_token id = common_sampler_sample(gsm, ctx, idx);
    common_sampler_accept(gsm, id, true);
}

struct llama_sampler * common_sampler_get(const struct common_sampler * gsm) {
    return gsm ? gsm->chain : nullptr;
}

struct common_sampler * common_sampler_clone(struct common_sampler * gsm) {
    return new common_sampler {
        gsm->params,
        gsm->grmr    ? llama_sampler_clone(gsm->grmr)    : nullptr,
        gsm->rbudget ? llama_sampler_clone(gsm->rbudget) : nullptr,
        gsm->chain   ? llama_sampler_clone(gsm->chain)   : nullptr,
        gsm->prev,
        gsm->cur,
        gsm->cur_p,
        gsm->t_total_us
    };
}

void common_sampler_get_candidates(struct common_sampler * gsm, std::vector<llama_token_data> & out, bool sorted) {
    if (!gsm) return;
    if (sorted) {
        std::sort(gsm->cur_p.data, gsm->cur_p.data + gsm->cur_p.size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.logit > b.logit;
        });
    }
    out.assign(gsm->cur_p.data, gsm->cur_p.data + gsm->cur_p.size);
}

llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsm, bool sorted) {
    if (!gsm) return nullptr;
    if (sorted && !gsm->cur_p.sorted) {
        std::sort(gsm->cur_p.data, gsm->cur_p.data + gsm->cur_p.size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.logit > b.logit;
        });
        gsm->cur_p.sorted = true;
    }
    return &gsm->cur_p;
}

llama_token common_sampler_last(const struct common_sampler * gsm) {
    return gsm->prev.back();
}

void common_sampler_reset(struct common_sampler * gsm) {
    gsm->reset();
}

uint32_t common_sampler_get_seed(const struct common_sampler * gsm) {
    return gsm->params.seed;
}

// =================================================================
// SOBRECARGAS UNIFICADAS DE SAMPLE AND ACCEPT N (ÚNICAS Y SIN DUPLICADOS)
// =================================================================
std::vector<llama_token> common_sampler_sample_and_accept_n(
        struct common_sampler * gsmpl, 
        struct llama_context * ctx, 
        const std::vector<int> & idxs, 
        const llama_tokens & draft, 
        bool grammar_first) {
    std::vector<llama_token> result;
    if (!gsmpl || idxs.empty()) return result;

    for (size_t i = 0; i < idxs.size(); ++i) {
        llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);
        common_sampler_accept(gsmpl, id, true);
        result.push_back(id);

        if (i < draft.size() && id != draft[i]) {
            break;
        }
    }
    return result;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(
        struct common_sampler * gsmpl, 
        struct llama_context * ctx, 
        const std::vector<int> & idxs, 
        bool grammar_first) {
    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, {}, grammar_first);
}

// =================================================================
// 4. PARSEO DE TIPOS, MÉTRICAS Y HELPERS PARA ARM64 / M1
// =================================================================
char common_sampler_type_to_chr(enum common_sampler_type type) {
    switch (type) {
        case COMMON_SAMPLER_TYPE_DRY:         return 'd';
        case COMMON_SAMPLER_TYPE_TOP_K:       return 'k';
        case COMMON_SAMPLER_TYPE_TOP_P:       return 'p';
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return 's';
        case COMMON_SAMPLER_TYPE_MIN_P:       return 'm';
        case COMMON_SAMPLER_TYPE_XTC:         return 'x';
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return 'y';
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return 't';
        case COMMON_SAMPLER_TYPE_INFILL:      return 'i';
        case COMMON_SAMPLER_TYPE_PENALTIES:   return 'c';
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return 'a';
        default:                              return '?';
    }
}

static enum common_sampler_type common_sampler_type_from_chr(char c) {
    switch (c) {
        case 'd': return COMMON_SAMPLER_TYPE_DRY;
        case 'k': return COMMON_SAMPLER_TYPE_TOP_K;
        case 'p': return COMMON_SAMPLER_TYPE_TOP_P;
        case 's': return COMMON_SAMPLER_TYPE_TOP_N_SIGMA;
        case 'm': return COMMON_SAMPLER_TYPE_MIN_P;
        case 'x': return COMMON_SAMPLER_TYPE_XTC;
        case 'y': return COMMON_SAMPLER_TYPE_TYPICAL_P;
        case 't': return COMMON_SAMPLER_TYPE_TEMPERATURE;
        case 'i': return COMMON_SAMPLER_TYPE_INFILL;
        case 'c': return COMMON_SAMPLER_TYPE_PENALTIES;
        case 'a': return COMMON_SAMPLER_TYPE_ADAPTIVE_P;
        default:  return COMMON_SAMPLER_TYPE_NONE;
    }
}

std::string common_sampler_type_to_str(enum common_sampler_type type) {
    switch (type) {
        case COMMON_SAMPLER_TYPE_DRY:         return "dry";
        case COMMON_SAMPLER_TYPE_TOP_K:       return "top_k";
        case COMMON_SAMPLER_TYPE_TOP_P:       return "top_p";
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return "top_n_sigma";
        case COMMON_SAMPLER_TYPE_MIN_P:       return "min_p";
        case COMMON_SAMPLER_TYPE_XTC:         return "xtc";
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return "typical_p";
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return "temperature";
        case COMMON_SAMPLER_TYPE_INFILL:      return "infill";
        case COMMON_SAMPLER_TYPE_PENALTIES:   return "penalties";
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return "adaptive_p";
        default:                              return "unknown";
    }
}

std::vector<enum common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names) {
    std::unordered_map<std::string, enum common_sampler_type> name_to_type = {
        {"dry",         COMMON_SAMPLER_TYPE_DRY},
        {"top_k",       COMMON_SAMPLER_TYPE_TOP_K},
        {"top_p",       COMMON_SAMPLER_TYPE_TOP_P},
        {"top_n_sigma", COMMON_SAMPLER_TYPE_TOP_N_SIGMA},
        {"min_p",       COMMON_SAMPLER_TYPE_MIN_P},
        {"xtc",         COMMON_SAMPLER_TYPE_XTC},
        {"typical_p",   COMMON_SAMPLER_TYPE_TYPICAL_P},
        {"temperature", COMMON_SAMPLER_TYPE_TEMPERATURE},
        {"infill",      COMMON_SAMPLER_TYPE_INFILL},
        {"penalties",   COMMON_SAMPLER_TYPE_PENALTIES},
        {"adaptive_p",  COMMON_SAMPLER_TYPE_ADAPTIVE_P},
    };

    std::vector<enum common_sampler_type> samplers;
    samplers.reserve(names.size());

    for (const auto & name : names) {
        auto it = name_to_type.find(name);
        if (it != name_to_type.end()) {
            samplers.push_back(it->second);
        } else {
            LOG_WRN("%s: unknown sampler name: %s\n", __func__, name.c_str());
        }
    }

    return samplers;
}

std::vector<enum common_sampler_type> common_sampler_types_from_chars(const std::string & chars) {
    std::vector<enum common_sampler_type> samplers;
    samplers.reserve(chars.size());

    for (const char c : chars) {
        enum common_sampler_type type = common_sampler_type_from_chr(c);
        if (type != COMMON_SAMPLER_TYPE_NONE) {
            samplers.push_back(type);
        } else {
            LOG_WRN("%s: unknown sampler char: %c\n", __func__, c);
        }
    }

    return samplers;
}

static std::string common_sampler_type_to_str_list(const std::vector<enum common_sampler_type> & samplers) {
    std::string result;
    for (size_t i = 0; i < samplers.size(); ++i) {
        if (i > 0) result += ";";
        result += common_sampler_type_to_str(samplers[i]);
    }
    return result;
}

static uint64_t common_sampler_get_time_us(const struct common_sampler * gsm) {
    return gsm->t_total_us;
}

void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl) {
    if (ctx) {
        llama_perf_context_print(ctx);
    }
    if (gsmpl && gsmpl->chain) {
        llama_perf_sampler_print(gsmpl->chain);
    }
}

// FIX DE SÍMBOLOS PARA COMPILACIÓN EN APPLE SILICON (ARM64 / M1)
std::string common_sampler_print(const struct common_sampler * gsmpl) {
    if (!gsmpl) return "";
    return gsmpl->params.print();
}
void common_sampler_reasoning_budget_force(struct common_sampler * gsmpl) {
    if (gsmpl && gsmpl->rbudget) {
        common_reasoning_budget_force(gsmpl->rbudget);
    }
}
std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx, int n) {
    if (!gsmpl || !ctx) return "";

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::string result;
    const size_t sz = gsmpl->prev.size();
    const size_t start = sz > (size_t)n ? sz - n : 0;

    for (size_t i = start; i < sz; ++i) {
        const llama_token id = gsmpl->prev.rat(sz - 1 - i);
        result += common_token_to_piece(vocab, id, true);
    }

    return result;
}