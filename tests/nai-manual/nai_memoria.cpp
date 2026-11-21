// [NAI] Memoria reconstructiva por regiones (v2).
//
// Los pesos del modelo son la "corteza": conocimiento general que ya está ahí.
// La memoria de la sesión es el "hipocampo": tamaño FIJO, organizada por regiones.
// Cada recuerdo (una oración de la conversación) se guarda en uno de tres niveles:
//   exacto    -> solo la SORPRESA (lo que el modelo no podía predecir); se reconstruye
//                el texto idéntico volviendo a recorrer el mismo camino con el modelo.
//   fragmento -> solo los tokens más sorprendentes; el modelo rellena el sentido.
//   pista     -> unas pocas palabras clave para volver a llegar al recuerdo.
// Cuando la memoria se llena, lo menos importante baja de nivel ("olvidar" sin borrar).
// Importancia = peso de la región x refuerzo por uso. Las regiones prioritarias
// (conciencia, ética, ciencia, técnica, arte...) pesan más y ceden al final.
// Un concepto activa recuerdos de regiones distintas y el modelo los cruza.
//
// Uso:
//   nai_memoria -m modelo.gguf [-c "lápiz"] [--memoria BYTES] [-r regiones.txt]
//               [--escribir-regiones regiones.txt] [-f recuerdos.txt] [-t 4]
//   regiones.txt: líneas "peso|tema|semilla" (peso 1 = general, 3 = prioritario)
//   recuerdos.txt: una oración por línea

#include "llama.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

struct region {
    int peso;
    std::string nombre, semilla;
};

static std::vector<region> regiones_por_defecto() {
    return {
        { 3, "conciencia",           "Darse cuenta de uno mismo, la experiencia interior y el estar despierto al presente" },
        { 3, "ética",                "Distinguir lo correcto de lo incorrecto, la responsabilidad y el respeto a los demás" },
        { 3, "espiritualidad",       "La búsqueda de sentido, lo sagrado, la contemplación y la conexión con algo mayor" },
        { 3, "empatía",              "Comprender y acompañar lo que siente otra persona" },
        { 3, "psique",               "La mente, las emociones, los sueños, los miedos y la conducta humana" },
        { 3, "filosofía",            "Preguntas sobre la existencia, la identidad, el conocimiento y la verdad" },
        { 3, "lógica",               "Razonar con premisas y conclusiones, deducir y detectar contradicciones" },
        { 3, "matemáticas",          "Números, demostraciones, ecuaciones, geometría y estructuras abstractas" },
        { 3, "física",               "Fuerzas, energía, movimiento, luz, materia y las leyes de la naturaleza" },
        { 3, "química",              "Composición de la materia, reacciones, enlaces y transformación de sustancias" },
        { 3, "biología",             "Los seres vivos, las células, las plantas, los animales y cómo funcionan" },
        { 3, "ingeniería",           "Diseñar y construir soluciones con restricciones de costo, material y seguridad" },
        { 3, "mecánica",             "Máquinas, palancas, engranes, motores y transmisión de fuerza y movimiento" },
        { 3, "aerodinámica",         "Cómo fluye el aire alrededor de los cuerpos: sustentación, resistencia y vuelo" },
        { 3, "ciberseguridad",       "Proteger sistemas, datos y personas frente a ataques, contraseñas y cifrado" },
        { 3, "supervivencia",        "Obtener agua, refugio, fuego y alimento, y orientarse en condiciones adversas" },
        { 3, "agricultura",          "Cultivar la tierra: suelo, semillas, agua, ciclos de siembra y cosecha" },
        { 3, "arte",                 "Expresar ideas y emociones con la forma, el color, el sonido o la palabra" },
        { 3, "medicina tradicional", "Remedios y prácticas de salud transmitidos por generaciones en distintas culturas" },
        { 1, "finanzas personales",  "Dinero, ahorro, cuentas bancarias, pagos y préstamos" },
        { 1, "cocina",               "Recetas, ingredientes y preparación de comida" },
        { 1, "compras",              "Comprar productos, tiendas y precios" },
        { 1, "deportes",             "Partidos, equipos, jugadores y competencias" },
        { 1, "trámites",             "Documentos, citas y gestiones administrativas" },
        { 1, "entretenimiento",      "Series, películas, música y tiempo libre" },
    };
}

static std::vector<std::string> recuerdos_por_defecto() {
    return {
        "Darse cuenta de que uno está pensando es distinto de simplemente pensar; esa mirada interior es la conciencia.",
        "Antes de usar los datos de un cliente para entrenar un sistema, hay que preguntarse si él estaría de acuerdo.",
        "Para Rosa, caminar en silencio por el cerro al amanecer es una forma de oración sin palabras.",
        "Cuando Pedro perdió su trabajo, lo que más le ayudó fue que su hermana lo escuchara sin darle consejos.",
        "El miedo que no se nombra suele aparecer en los sueños como persecuciones o caídas.",
        "Si cambias cada tabla de un barco una por una, ¿sigue siendo el mismo barco al final?",
        "Si todos los mensajes firmados son auténticos y este mensaje no es auténtico, entonces no está firmado.",
        "El profesor Ruiz resolvió la demostración a lápiz, dividiendo el problema en dos casos.",
        "Ruiz dice que el papel es una herramienta para plasmar lo que piensa antes de saber si es verdad.",
        "Un objeto cae más lento en el aire que en el vacío porque la resistencia del aire se opone a su movimiento.",
        "Al mezclar bicarbonato con vinagre se libera dióxido de carbono, por eso aparecen burbujas.",
        "Las raíces de una planta crecen hacia donde hay más humedad, aunque tengan que rodear una piedra.",
        "El puente se diseñó para soportar el triple de la carga máxima esperada, por seguridad.",
        "Una palanca larga permite mover una roca pesada con poca fuerza, a cambio de recorrer más distancia.",
        "El ala de un avión es más curva por arriba, y esa forma ayuda a generar sustentación.",
        "Una contraseña hecha de cuatro palabras al azar es más difícil de adivinar que una corta con símbolos.",
        "Si te pierdes en el monte, primero busca refugio y agua; la comida puede esperar varios días.",
        "Rotar el maíz con frijol cada temporada devuelve nitrógeno al suelo.",
        "Para Lucía el lápiz es donde empieza todo: traza la sombra del rostro antes de pintar el retrato.",
        "A veces, en el margen del cuaderno, Lucía escribe a lápiz un poema sobre la persona que dibuja.",
        "La abuela preparaba té de manzanilla con miel para calmar el dolor de estómago.",
        "Juan abrió una cuenta de ahorro el 3 de marzo con 2,500 pesos y deposita 400 cada quincena.",
        "Decidió dejar la mitad de sus ahorros a plazo fijo hasta diciembre.",
        "La receta de la abuela no tiene medidas exactas: la harina se mide a ojo.",
        "El sábado compraron un refrigerador nuevo porque el anterior hacía mucho ruido.",
        "El equipo local perdió el domingo por un gol en el último minuto.",
        "Para renovar el pasaporte hay que sacar cita en línea y llevar una identificación.",
        "Anoche vieron una serie de detectives hasta muy tarde.",
    };
}

static std::vector<region> leer_regiones(const std::string & ruta) {
    std::vector<region> res;
    std::ifstream f(ruta);
    std::string l;
    while (std::getline(f, l)) {
        if (l.empty() || l[0] == '#') continue;
        const size_t a = l.find('|'), b = l.find('|', a + 1);
        if (a == std::string::npos || b == std::string::npos) continue;
        res.push_back({ std::max(1, atoi(l.substr(0, a).c_str())), l.substr(a + 1, b - a - 1), l.substr(b + 1) });
    }
    return res;
}

static void escribir_regiones(const std::string & ruta, const std::vector<region> & rs) {
    std::ofstream f(ruta);
    f << "# peso|tema|semilla   (peso 1 = general, 3 = prioritario; más alto = se conserva más)\n";
    for (auto & r : rs) f << r.peso << "|" << r.nombre << "|" << r.semilla << "\n";
}

static std::vector<std::string> leer_recuerdos(const std::string & ruta) {
    std::vector<std::string> res;
    std::ifstream f(ruta);
    std::string l;
    while (std::getline(f, l)) if (l.size() > 8) res.push_back(l);
    return res;
}

static std::vector<llama_token> tokenizar(const llama_vocab * vocab, const std::string & s, bool bos) {
    std::vector<llama_token> t(s.size() + 16);
    const int n = llama_tokenize(vocab, s.c_str(), (int) s.size(), t.data(), (int) t.size(), bos, true);
    t.resize(std::max(0, n));
    return t;
}

static std::string detokenizar(const llama_vocab * vocab, const std::vector<llama_token> & toks) {
    std::string s;
    char buf[256];
    for (auto t : toks) {
        const int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, false);
        if (n > 0) s.append(buf, n);
    }
    return s;
}
// ---------------------------------------------------------------------------
// Probabilidades cuantizadas (deterministas) y rANS de 64 bits
// ---------------------------------------------------------------------------
static const uint32_t ESCALA_BITS = 24;
static const uint32_t ESCALA      = 1u << ESCALA_BITS;
static const uint64_t RANS_L      = 1ull << 31;

static void cdf_desde_logits(const float * logits, int n_vocab, std::vector<uint32_t> & cum) {
    double lmax = logits[0];
    int    vmax = 0;
    for (int v = 1; v < n_vocab; ++v) if (logits[v] > lmax) { lmax = logits[v]; vmax = v; }
    std::vector<double> e(n_vocab);
    double z = 0.0;
    for (int v = 0; v < n_vocab; ++v) { e[v] = std::exp((double) logits[v] - lmax); z += e[v]; }
    const double resto = (double) (ESCALA - (uint32_t) n_vocab);
    std::vector<uint32_t> fr(n_vocab);
    uint64_t suma = 0;
    for (int v = 0; v < n_vocab; ++v) { fr[v] = 1u + (uint32_t) std::floor(e[v] / z * resto); suma += fr[v]; }
    fr[vmax] += (uint32_t) (ESCALA - suma);
    cum.resize(n_vocab + 1);
    cum[0] = 0;
    for (int v = 0; v < n_vocab; ++v) cum[v + 1] = cum[v] + fr[v];
}

struct rans_enc {
    uint64_t x = RANS_L;
    std::vector<uint32_t> palabras;
    void poner(uint32_t inicio, uint32_t freq) {
        const uint64_t x_max = ((RANS_L >> ESCALA_BITS) << 32) * freq;
        if (x >= x_max) { palabras.push_back((uint32_t) x); x >>= 32; }
        x = ((x / freq) << ESCALA_BITS) + (x % freq) + inicio;
    }
    std::vector<uint32_t> terminar() {
        palabras.push_back((uint32_t) (x >> 32));
        palabras.push_back((uint32_t) x);
        std::reverse(palabras.begin(), palabras.end());
        return palabras;
    }
};

struct rans_dec {
    uint64_t x = 0;
    const std::vector<uint32_t> * p = nullptr;
    size_t i = 0;
    void iniciar(const std::vector<uint32_t> & w) { p = &w; x = (uint64_t) w[0] | ((uint64_t) w[1] << 32); i = 2; }
    uint32_t mirar() const { return (uint32_t) (x & (ESCALA - 1)); }
    void avanzar(uint32_t inicio, uint32_t freq) {
        x = freq * (x >> ESCALA_BITS) + (x & (ESCALA - 1)) - inicio;
        if (x < RANS_L && i < p->size()) x = (x << 32) | (*p)[i++];
    }
};

static bool evaluar_uno(llama_context * ctx, llama_token t) {
    return llama_decode(ctx, llama_batch_get_one(&t, 1)) == 0;
}

struct comprimido {
    std::vector<uint32_t> palabras;
    int n_tokens = 0;
    std::vector<double> sorpresa_bits; // sorpresa_bits[i] corresponde a toks[i+1]
};

static comprimido comprimir(llama_context * ctx, const llama_vocab * vocab, const std::vector<llama_token> & toks) {
    comprimido c;
    const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_memory_clear(llama_get_memory(ctx), true);
    std::vector<uint32_t> cum;
    std::vector<std::pair<uint32_t, uint32_t>> simbolos;
    evaluar_uno(ctx, toks[0]);
    for (size_t i = 1; i < toks.size(); ++i) {
        cdf_desde_logits(llama_get_logits_ith(ctx, -1), n_vocab, cum);
        const uint32_t ini = cum[toks[i]], fr = cum[toks[i] + 1] - cum[toks[i]];
        simbolos.push_back({ ini, fr });
        c.sorpresa_bits.push_back(-std::log2((double) fr / ESCALA));
        if (i + 1 < toks.size()) evaluar_uno(ctx, toks[i]);
    }
    rans_enc enc;
    for (size_t k = simbolos.size(); k-- > 0;) enc.poner(simbolos[k].first, simbolos[k].second);
    c.palabras = enc.terminar();
    c.n_tokens = (int) toks.size();
    return c;
}

static std::vector<llama_token> reconstruir(llama_context * ctx, const llama_vocab * vocab, const comprimido & c, llama_token primero) {
    const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_memory_clear(llama_get_memory(ctx), true);
    std::vector<llama_token> toks = { primero };
    std::vector<uint32_t> cum;
    rans_dec dec;
    dec.iniciar(c.palabras);
    evaluar_uno(ctx, primero);
    for (int i = 1; i < c.n_tokens; ++i) {
        cdf_desde_logits(llama_get_logits_ith(ctx, -1), n_vocab, cum);
        const uint32_t m = dec.mirar();
        const int v = (int) (std::upper_bound(cum.begin(), cum.end(), m) - cum.begin()) - 1;
        dec.avanzar(cum[v], cum[v + 1] - cum[v]);
        toks.push_back(v);
        if (i + 1 < c.n_tokens) evaluar_uno(ctx, v);
    }
    return toks;
}
// ---------------------------------------------------------------------------
// Generación, comparación de palabras y embeddings
// ---------------------------------------------------------------------------
static std::string generar(llama_context * ctx, const llama_model * model, const std::string & pregunta, int n_max) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::string prompt = pregunta;
    if (const char * tmpl = llama_model_chat_template(model, nullptr)) {
        llama_chat_message msg = { "user", pregunta.c_str() };
        std::vector<char> buf(pregunta.size() * 4 + 4096);
        const int n = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), (int) buf.size());
        if (n > 0 && n <= (int) buf.size()) prompt.assign(buf.data(), n);
    }
    llama_memory_clear(llama_get_memory(ctx), true);
    std::vector<llama_token> toks = tokenizar(vocab, prompt, true);
    if (llama_decode(ctx, llama_batch_get_one(toks.data(), (int) toks.size())) != 0) return "";
    std::vector<llama_token> out;
    for (int i = 0; i < n_max; ++i) {
        const float * l = llama_get_logits_ith(ctx, -1);
        llama_token t = 0;
        for (int v = 1; v < n_vocab; ++v) if (l[v] > l[t]) t = v;
        if (llama_vocab_is_eog(vocab, t)) break;
        out.push_back(t);
        if (!evaluar_uno(ctx, t)) break;
    }
    return detokenizar(vocab, out);
}

static std::vector<std::string> palabras(const std::string & s) {
    std::vector<std::string> res;
    std::string w;
    for (unsigned char ch : s) {
        if (std::isalnum(ch) || ch >= 0x80) { w += (char) std::tolower(ch); }
        else if (!w.empty()) { res.push_back(w); w.clear(); }
    }
    if (!w.empty()) res.push_back(w);
    return res;
}

static double recuperacion(const std::string & original, const std::string & recon) {
    std::set<std::string> a, b;
    for (auto & w : palabras(original)) if (w.size() > 2) a.insert(w);
    for (auto & w : palabras(recon))    b.insert(w);
    if (a.empty()) return 0.0;
    int n = 0;
    for (auto & w : a) n += b.count(w);
    return 100.0 * n / a.size();
}

static std::vector<float> embedding(llama_context * ctx_emb, const llama_vocab * vocab, const std::string & s, int n_embd) {
    llama_memory_clear(llama_get_memory(ctx_emb), true);
    std::vector<llama_token> toks = tokenizar(vocab, s, true);
    llama_batch b = llama_batch_init((int) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i] = toks[i]; b.pos[i] = (llama_pos) i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = true;
    }
    b.n_tokens = (int) toks.size();
    std::vector<float> e(n_embd, 0.0f);
    if (llama_decode(ctx_emb, b) == 0) {
        if (const float * p = llama_get_embeddings_seq(ctx_emb, 0)) e.assign(p, p + n_embd);
    }
    llama_batch_free(b);
    return e;
}

static void normalizar(std::vector<float> & e) {
    double n = 0;
    for (float v : e) n += (double) v * v;
    n = std::sqrt(n);
    if (n > 0) for (float & v : e) v = (float) (v / n);
}

static float coseno(const std::vector<float> & a, const std::vector<float> & b) {
    float s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

// ---------------------------------------------------------------------------
// Memoria por regiones
// ---------------------------------------------------------------------------
enum nivel { EXACTO = 0, FRAGMENTO = 1, PISTA = 2 };
static const char * NOMBRE_NIVEL[] = { "exacto", "fragmento", "pista" };

struct recuerdo {
    std::string texto;
    std::vector<llama_token> toks;
    comprimido comp;
    std::vector<float> e;
    int region = 0, region2 = -1;
    int usos = 0;
    int nivel = EXACTO;
    size_t costo[3] = { 0, 0, 0 };
    std::string fragmento, pista;
    double densidad = 0; // bits de sorpresa por token
};

// Concepto -> activación de cada recuerdo (directa + propagada desde otra región).
static std::vector<float> activar(const std::vector<recuerdo> & rs, const std::vector<float> & eq) {
    std::vector<float> directo(rs.size()), activ(rs.size());
    for (size_t i = 0; i < rs.size(); ++i) directo[i] = coseno(eq, rs[i].e);
    for (size_t i = 0; i < rs.size(); ++i) {
        float puente = 0;
        for (size_t j = 0; j < rs.size(); ++j) {
            if (j == i || rs[j].region == rs[i].region) continue;
            puente = std::max(puente, directo[j] * coseno(rs[j].e, rs[i].e));
        }
        activ[i] = directo[i] + 0.5f * puente;
    }
    return activ;
}

// Los 3 recuerdos más activados, de regiones distintas (fuerza el cruce).
static std::vector<int> elegir_cruce(const std::vector<recuerdo> & rs, const std::vector<float> & activ) {
    std::vector<int> orden(rs.size());
    for (size_t i = 0; i < orden.size(); ++i) orden[i] = (int) i;
    std::sort(orden.begin(), orden.end(), [&](int a, int b) { return activ[a] > activ[b]; });
    std::vector<int> res;
    std::set<int> usadas;
    for (int i : orden) {
        if (usadas.count(rs[i].region)) continue;
        res.push_back(i);
        usadas.insert(rs[i].region);
        if (res.size() == 3) break;
    }
    return res;
}
int main(int argc, char ** argv) {
    std::string ruta, ruta_regiones, ruta_recuerdos, escribir, concepto = "lápiz";
    long memoria = 0;
    int n_threads = 4;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "-m" && i + 1 < argc) ruta = argv[++i];
        else if (a == "-r" && i + 1 < argc) ruta_regiones = argv[++i];
        else if (a == "-f" && i + 1 < argc) ruta_recuerdos = argv[++i];
        else if (a == "-c" && i + 1 < argc) concepto = argv[++i];
        else if (a == "-t" && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (a == "--memoria" && i + 1 < argc) memoria = atol(argv[++i]);
        else if (a == "--escribir-regiones" && i + 1 < argc) escribir = argv[++i];
    }
    std::vector<region> regs = ruta_regiones.empty() ? regiones_por_defecto() : leer_regiones(ruta_regiones);
    if (!escribir.empty()) {
        escribir_regiones(escribir, regs);
        printf("regiones escritas en %s (edita pesos y semillas, y úsalas con -r)\n", escribir.c_str());
        if (ruta.empty()) return 0;
    }
    if (ruta.empty()) {
        fprintf(stderr, "uso: %s -m modelo.gguf [-c concepto] [--memoria BYTES] [-r regiones.txt] "
                        "[--escribir-regiones regiones.txt] [-f recuerdos.txt] [-t 4]\n", argv[0]);
        return 1;
    }
    const std::vector<std::string> textos = ruta_recuerdos.empty() ? recuerdos_por_defecto() : leer_recuerdos(ruta_recuerdos);
    if (regs.empty() || textos.empty()) { fprintf(stderr, "sin regiones o sin recuerdos\n"); return 1; }

    llama_backend_init();
    llama_log_set([](ggml_log_level lvl, const char * txt, void *) { if (lvl >= GGML_LOG_LEVEL_ERROR) fputs(txt, stderr); }, nullptr);
    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = [](float, void *) { return true; };
    llama_model * model = llama_model_load_from_file(ruta.c_str(), mp);
    if (!model) { fprintf(stderr, "no se pudo cargar el modelo\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = 2048; cp.n_threads = n_threads; cp.n_threads_batch = n_threads;
    llama_context * ctx = llama_init_from_model(model, cp);
    llama_context_params ce = cp;
    ce.embeddings = true;
    ce.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    llama_context * ctx_emb = llama_init_from_model(model, ce);
    if (!ctx || !ctx_emb) { fprintf(stderr, "no se pudieron crear los contextos\n"); return 1; }
    const int head_dim = n_embd / llama_model_n_head(model);
    const double kv_por_token = (double) llama_model_n_layer(model) * 2.0 * llama_model_n_head_kv(model) * head_dim * 2.0;

    // ---- embeddings centrados (quita la dirección común para distinguir mejor los temas)
    std::vector<std::vector<float>> e_reg, e_rec;
    for (auto & r : regs)   e_reg.push_back(embedding(ctx_emb, vocab, "Tema: " + r.nombre + ". " + r.semilla, n_embd));
    for (auto & t : textos) e_rec.push_back(embedding(ctx_emb, vocab, t, n_embd));
    std::vector<float> media(n_embd, 0.0f);
    for (auto * grupo : { &e_reg, &e_rec }) for (auto & e : *grupo) for (int d = 0; d < n_embd; ++d) media[d] += e[d];
    for (auto & m : media) m /= (float) (e_reg.size() + e_rec.size());
    auto centrar = [&](std::vector<float> e) { for (int d = 0; d < n_embd; ++d) e[d] -= media[d]; normalizar(e); return e; };
    for (auto & e : e_reg) e = centrar(e);

    // ---- 1. clasificar y comprimir
    std::vector<recuerdo> rs(textos.size());
    int n_identicos = 0;
    double total_kv = 0, total_texto = 0;
    for (size_t i = 0; i < textos.size(); ++i) {
        recuerdo & r = rs[i];
        r.texto = textos[i];
        r.e = centrar(e_rec[i]);
        float best = -2, second = -2;
        for (size_t k = 0; k < regs.size(); ++k) {
            const float s = coseno(r.e, e_reg[k]);
            if (s > best) { second = best; r.region2 = r.region; best = s; r.region = (int) k; }
            else if (s > second) { second = s; r.region2 = (int) k; }
        }
        if (best - second > 0.05f) r.region2 = -1; // solo doble pertenencia si está muy cerca de dos regiones
        r.toks = tokenizar(vocab, r.texto, true);
        r.comp = comprimir(ctx, vocab, r.toks);
        n_identicos += reconstruir(ctx, vocab, r.comp, r.toks[0]) == r.toks;
        const auto & s = r.comp.sorpresa_bits;
        double bits = 0;
        for (double b : s) bits += b;
        r.densidad = s.empty() ? 0 : bits / s.size();
        std::vector<int> idx(s.size());
        for (size_t j = 0; j < idx.size(); ++j) idx[j] = (int) j;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return s[a] > s[b]; });
        const int n_frag = std::max(1, (int) std::round(0.35 * s.size()));
        std::vector<int> keep(idx.begin(), idx.begin() + std::min<int>(n_frag, (int) idx.size()));
        std::sort(keep.begin(), keep.end());
        int prev = -2;
        for (int j : keep) {
            if (j != prev + 1 && !r.fragmento.empty()) r.fragmento += " …";
            r.fragmento += detokenizar(vocab, { r.toks[j + 1] });
            prev = j;
        }
        std::vector<int> top(idx.begin(), idx.begin() + std::min<int>(3, (int) idx.size()));
        std::sort(top.begin(), top.end());
        for (int j : top) r.pista += detokenizar(vocab, { r.toks[j + 1] });
        r.costo[EXACTO]    = r.comp.palabras.size() * 4 + 2;
        r.costo[FRAGMENTO] = r.fragmento.size() + 2;
        r.costo[PISTA]     = r.pista.size() + 2;
        total_kv += kv_por_token * r.toks.size();
        total_texto += r.texto.size();
    }

    printf("1) REGIONES: %zu recuerdos clasificados en %zu regiones\n\n", rs.size(), regs.size());
    for (size_t k = 0; k < regs.size(); ++k) {
        std::vector<int> mios;
        for (size_t i = 0; i < rs.size(); ++i) if (rs[i].region == (int) k || rs[i].region2 == (int) k) mios.push_back((int) i);
        if (mios.empty()) continue;
        printf("  %-21s peso %d  %2zu  ej.: %.60s\n", regs[k].nombre.c_str(), regs[k].peso, mios.size(), rs[mios[0]].texto.c_str());
    }

    size_t total_exacto = 0;
    for (auto & r : rs) total_exacto += r.costo[EXACTO];
    printf("\n2) MEMORIA EXACTA: %d/%zu recuerdos reconstruidos idénticos\n", n_identicos, rs.size());
    printf("   KV cache %.0f KB | texto %.0f B | solo sorpresa %zu B  (%.0fx menos que KV, %.1fx menos que texto)\n\n",
           total_kv / 1024, total_texto, total_exacto, total_kv / total_exacto, total_texto / total_exacto);
    // ---- 3. refuerzo por uso: consultar conceptos sube la importancia de lo que activan
    const std::vector<std::string> consultas = { concepto, "energía", "cuidar a otros", "proteger" };
    printf("3) REFUERZO: consultas que activan recuerdos (cada uso suma importancia)\n\n");
    for (auto & q : consultas) {
        const auto activ = activar(rs, centrar(embedding(ctx_emb, vocab, q, n_embd)));
        const auto el = elegir_cruce(rs, activ);
        printf("  \"%s\" ->", q.c_str());
        for (int i : el) { rs[i].usos++; printf(" [%s]", regs[rs[i].region].nombre.c_str()); }
        printf("\n");
    }

    // ---- 4. presupuesto fijo: degradar lo menos importante hasta caber
    if (memoria <= 0) memoria = (long) (total_exacto / 2);
    auto importancia = [&](const recuerdo & r) { return regs[r.region].peso * (1.0 + 0.5 * r.usos) + r.densidad / 100.0; };
    auto total = [&]() { size_t t = 0; for (auto & r : rs) t += r.costo[r.nivel]; return t; };
    std::vector<std::string> degradaciones;
    while ((long) total() > memoria) {
        int pick = -1;
        for (size_t i = 0; i < rs.size(); ++i) {
            if (rs[i].nivel == PISTA) continue;
            if (pick < 0 || importancia(rs[i]) < importancia(rs[pick])) pick = (int) i;
        }
        if (pick < 0) break;
        rs[pick].nivel++;
        degradaciones.push_back(regs[rs[pick].region].nombre + " -> " + NOMBRE_NIVEL[rs[pick].nivel]);
    }
    printf("\n4) MEMORIA FIJA de %ld B (todo exacto ocuparía %zu B); usado %zu B%s\n\n", memoria, total_exacto, total(),
           (long) total() > memoria ? "  [NO CABE: presupuesto demasiado chico]" : "");
    printf("  %-21s %4s %6s %9s %6s %7s\n", "región", "peso", "exacto", "fragmento", "pista", "bytes");
    for (size_t k = 0; k < regs.size(); ++k) {
        int n[3] = { 0, 0, 0 };
        size_t b = 0;
        bool hay = false;
        for (auto & r : rs) if (r.region == (int) k) { n[r.nivel]++; b += r.costo[r.nivel]; hay = true; }
        if (hay) printf("  %-21s %4d %6d %9d %6d %6zu\n", regs[k].nombre.c_str(), regs[k].peso, n[0], n[1], n[2], b);
    }
    printf("\n  orden en que se fue \"olvidando\" (primeros 10):");
    for (size_t i = 0; i < degradaciones.size() && i < 10; ++i) printf("%s %s", i ? "," : "", degradaciones[i].c_str());
    printf("\n");

    // ---- 5. reconstruir desde niveles degradados
    printf("\n5) VOLVER A LLEGAR: reconstrucción desde fragmentos y pistas\n\n");
    int mostrados = 0;
    for (auto & r : rs) {
        if (r.nivel == EXACTO || mostrados >= 4) continue;
        const std::string base = r.nivel == FRAGMENTO ? r.fragmento : r.pista;
        const std::string recon = generar(ctx, model,
            "Estos son restos de un recuerdo en español. Reconstruye la oración original completa, sin explicar nada:\n" + base,
            (int) r.toks.size() + 20);
        printf("  [%s, %s] guardado: \"%s\"\n    reconstruido (%4.1f%% de palabras): %s\n    original: %s\n",
               regs[r.region].nombre.c_str(), NOMBRE_NIVEL[r.nivel], base.c_str(), recuperacion(r.texto, recon), recon.c_str(), r.texto.c_str());
        mostrados++;
    }

    // ---- 6. cruzar fragmentos de regiones distintas
    printf("\n6) CRUZAR: el concepto \"%s\" activa recuerdos de regiones distintas\n\n", concepto.c_str());
    const auto activ = activar(rs, centrar(embedding(ctx_emb, vocab, concepto, n_embd)));
    std::string ideas;
    for (int i : elegir_cruce(rs, activ)) {
        printf("  [%s] activación %.3f  %s\n", regs[rs[i].region].nombre.c_str(), activ[i], rs[i].texto.c_str());
        ideas += "- " + rs[i].texto + "\n";
    }
    const std::string nueva = generar(ctx, model,
        "Estas ideas vienen de áreas que no tienen relación entre sí:\n" + ideas +
        "Encuentra lo que tienen en común alrededor de \"" + concepto + "\" y propón una idea nueva que las conecte, "
        "en dos o tres oraciones.", 120);
    printf("\n  Idea nueva (cruce de fragmentos):\n  %s\n", nueva.c_str());

    llama_free(ctx_emb);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
