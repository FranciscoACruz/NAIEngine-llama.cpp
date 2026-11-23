// [NAI] Memoria reconstructiva por regiones (v3). Ver nai-memoria.h.
#include "nai-memoria.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>

namespace nai_mem {

// ---------------------------------------------------------------------------
// Regiones
// ---------------------------------------------------------------------------
std::vector<region> regiones_por_defecto() {
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
        { 1, "general",              "Saludos, datos personales y conversación cotidiana que no encaja en otro tema" },
    };
}

std::vector<region> leer_regiones(const std::string & ruta) {
    std::vector<region> res;
    std::ifstream f(ruta);
    std::string l;
    while (std::getline(f, l)) {
        if (l.empty() || l[0] == '#') continue;
        const size_t a = l.find('|'), b = l.find('|', a == std::string::npos ? a : a + 1);
        if (a == std::string::npos || b == std::string::npos) continue;
        res.push_back({ std::max(1, atoi(l.substr(0, a).c_str())), l.substr(a + 1, b - a - 1), l.substr(b + 1) });
        if (res.size() == 32) break; // la región se guarda en 5 bits
    }
    return res;
}

const char * nombre_nivel(int n) {
    static const char * N[] = { "exacto", "idea", "pista" };
    return n >= 0 && n < 3 ? N[n] : "?";
}

const char * nombre_rol(int r) {
    static const char * N[] = { "documento", "usuario", "asistente" };
    return r >= 0 && r < 3 ? N[r] : "?";
}

// ---------------------------------------------------------------------------
// Utilidades de texto
// ---------------------------------------------------------------------------
static uint32_t fnv32(const std::string & s) {
    uint32_t h = 2166136261u;
    for (unsigned char c : s) { h ^= c; h *= 16777619u; }
    return h;
}

static uint8_t check8(const std::string & s) {
    const uint32_t h = fnv32(s);
    return (uint8_t) (h ^ (h >> 8) ^ (h >> 16) ^ (h >> 24));
}

uint16_t hoy() {
    const int64_t t0 = 1767225600; // 2026-01-01T00:00:00Z
    const int64_t d  = ((int64_t) std::time(nullptr) - t0) / 86400;
    return (uint16_t) std::clamp<int64_t>(d, 0, 65535);
}

// Minúsculas y sin acentos (latín común en UTF-8).
static std::string normalizar(const std::string & w) {
    std::string r;
    for (size_t i = 0; i < w.size(); ++i) {
        const unsigned char c = (unsigned char) w[i];
        if (c < 0x80) { r += (char) std::tolower(c); continue; }
        if (c == 0xC3 && i + 1 < w.size()) {
            const unsigned char d = (unsigned char) w[++i];
            switch (d) {
                case 0xA1: case 0x81: case 0xA0: case 0x80: case 0xA4: case 0x84: case 0xA2: case 0x82: r += 'a'; break;
                case 0xA9: case 0x89: case 0xA8: case 0x88: case 0xAB: case 0x8B: case 0xAA: case 0x8A: r += 'e'; break;
                case 0xAD: case 0x8D: case 0xAC: case 0x8C: case 0xAF: case 0x8F: case 0xAE: case 0x8E: r += 'i'; break;
                case 0xB3: case 0x93: case 0xB2: case 0x92: case 0xB6: case 0x96: case 0xB4: case 0x94: r += 'o'; break;
                case 0xBA: case 0x9A: case 0xB9: case 0x99: case 0xBC: case 0x9C: case 0xBB: case 0x9B: r += 'u'; break;
                case 0xB1: case 0x91: r += 'n'; break;
                case 0xA7: case 0x87: r += 'c'; break;
                default: r += (char) c; r += (char) d; break;
            }
            continue;
        }
        r += (char) c;
    }
    return r;
}

static const std::set<std::string> & relleno() {
    static const std::set<std::string> s = {
        // español
        "para", "pero", "como", "porque", "cuando", "donde", "sobre", "entre", "hasta", "desde", "este", "esta",
        "estos", "estas", "esto", "aquel", "ella", "ellos", "ellas", "usted", "ustedes", "nosotros", "tiene",
        "tienen", "tener", "hace", "hacer", "hacen", "puede", "pueden", "poder", "debe", "deben", "sido", "estar",
        "esta", "estan", "estaba", "fueron", "seria", "solo", "tambien", "todo", "todos", "toda", "todas", "cada",
        "otro", "otra", "otros", "otras", "mismo", "misma", "algo", "alguien", "nada", "mucho", "mucha", "muchos",
        "poco", "pocos", "mas", "menos", "muy", "bien", "aqui", "ahi", "alla", "entonces", "luego", "despues",
        "antes", "ahora", "siempre", "nunca", "segun", "mientras", "aunque", "sino", "pues", "cual", "cuales",
        "quien", "quienes", "cuanto", "cuanta", "cuantos", "donde", "hola", "gracias", "favor", "porfavor",
        "dijo", "decir", "dice", "eres", "somos", "soy", "estoy", "esos", "esas", "eso", "ese", "esa", "usuario",
        "asistente", "respuesta", "pregunta", "claro", "puedo", "quiero", "quieres", "sabes", "saber",
        // inglés
        "that", "this", "with", "from", "have", "were", "what", "when", "where", "which", "there", "their",
        "about", "would", "could", "should", "these", "those", "your", "they", "them", "then", "than", "been",
        "will", "into", "just", "also", "some", "more", "most", "very", "only", "like", "here", "does",
    };
    return s;
}

uint16_t clave_palabra(const std::string & palabra) {
    std::string n = normalizar(palabra);
    if (n.size() < 4 || relleno().count(n)) return 0;
    bool digitos = std::all_of(n.begin(), n.end(), [](unsigned char c) { return std::isdigit(c); });
    if (n.size() > 6 && !digitos) n.resize(6); // raíz: "ahorros" y "ahorro" comparten clave
    uint32_t x = fnv32(n);
    x ^= x >> 15; x *= 0x2c1b3c6dU; x ^= x >> 12; // mezclar antes de quedarse con 16 bits
    const uint16_t h = (uint16_t) (x ^ (x >> 16));
    return h == 0 ? 1 : h;
}

// Palabras: letras y dígitos (incluye letras UTF-8); los signos ¿ ¡ « » y la puntuación
// tipográfica (“ ” — …) separan palabras.
static std::vector<std::string> palabras_de(const std::string & s) {
    std::vector<std::string> res;
    std::string w;
    auto cortar = [&]() { if (!w.empty()) { res.push_back(w); w.clear(); } };
    for (size_t i = 0; i < s.size(); ++i) {
        const unsigned char ch = (unsigned char) s[i];
        if (ch == 0xC2 && i + 1 < s.size()) { cortar(); ++i; continue; }                          // U+0080..U+00BF
        if (ch == 0xE2 && i + 2 < s.size() && (unsigned char) s[i + 1] == 0x80) { cortar(); i += 2; continue; } // U+2000..U+203F
        if (std::isalnum(ch) || ch >= 0x80) { w += (char) ch; }
        else cortar();
    }
    cortar();
    return res;
}

std::vector<uint16_t> claves_de(const std::string & texto) {
    std::vector<uint16_t> res;
    for (auto & w : palabras_de(texto)) {
        const uint16_t k = clave_palabra(w);
        if (k && std::find(res.begin(), res.end(), k) == res.end()) res.push_back(k);
    }
    return res;
}

static std::string colapsar(const std::string & s) {
    std::string r;
    bool esp = false;
    for (unsigned char c : s) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { esp = true; continue; }
        if (esp && !r.empty()) r += ' ';
        esp = false;
        r += (char) c;
    }
    return r;
}

// No cortar a la mitad de un carácter UTF-8.
static size_t corte_utf8(const std::string & s, size_t n) {
    if (n >= s.size()) return s.size();
    while (n > 0 && ((unsigned char) s[n] & 0xC0) == 0x80) --n;
    return n;
}

static std::string recortar(const std::string & s, size_t n) {
    return s.substr(0, corte_utf8(s, n));
}

static constexpr char SEP_TURNO = '\x1e';  // separador de registros ASCII: no aparece en texto normal

static std::vector<std::string> partir_turnos(const std::string & s) {
    std::vector<std::string> v;
    size_t a = 0;
    for (size_t b; (b = s.find(SEP_TURNO, a)) != std::string::npos; a = b + 1) v.push_back(s.substr(a, b - a));
    if (!s.empty()) v.push_back(s.substr(a));
    return v;
}

// Agrega un turno al anillo; descarta los más viejos hasta caber en max_turnos y max_bytes (cada lado).
static void anillo_turnos(memoria & m, const std::string & u, const std::string & a, int max_turnos, size_t max_bytes) {
    auto us = partir_turnos(m.previo_usuario), as = partir_turnos(m.previo_asistente);
    if (us.size() != as.size()) { us.clear(); as.clear(); }   // anillo inconsistente: se reinicia
    us.push_back(recortar(u, 1500));
    as.push_back(recortar(a, 1500));
    auto unir = [](const std::vector<std::string> & v, size_t desde) {
        std::string r;
        for (size_t i = desde; i < v.size(); ++i) { if (i > desde) r += SEP_TURNO; r += v[i]; }
        return r;
    };
    size_t d = us.size() > (size_t) std::max(1, max_turnos) ? us.size() - std::max(1, max_turnos) : 0;
    for (;; ++d) {
        m.previo_usuario   = unir(us, d);
        m.previo_asistente = unir(as, d);
        if (d + 1 >= us.size() || (m.previo_usuario.size() <= max_bytes && m.previo_asistente.size() <= max_bytes)) break;
    }
}

std::vector<std::string> trocear(const std::string & texto, size_t max_chars, size_t min_chars) {
    std::vector<std::string> parrafos, res;
    {
        std::string actual;
        std::istringstream in(texto);
        std::string l;
        while (std::getline(in, l)) {
            const bool vacia = l.find_first_not_of(" \t\r") == std::string::npos;
            if (vacia) { if (!actual.empty()) { parrafos.push_back(actual); actual.clear(); } continue; }
            actual += l;
            actual += '\n';
        }
        if (!actual.empty()) parrafos.push_back(actual);
    }
    for (auto & p0 : parrafos) {
        const std::string p = colapsar(p0);
        if (p.size() <= max_chars) { if (p.size() >= min_chars) res.push_back(p); continue; }
        // partir por oraciones
        std::string trozo;
        size_t ini = 0;
        while (ini < p.size()) {
            size_t fin = ini;
            while (fin < p.size()) {
                const char c = p[fin++];
                if ((c == '.' || c == '?' || c == '!' || c == ';') && (fin == p.size() || p[fin] == ' ')) break;
            }
            std::string oracion = p.substr(ini, fin - ini);
            ini = fin;
            while (oracion.size() > max_chars) { // oración enorme: cortar en un espacio
                size_t c = corte_utf8(oracion, max_chars);
                const size_t esp = oracion.rfind(' ', c);
                if (esp != std::string::npos && esp > max_chars / 2) c = esp;
                if (!trozo.empty()) { res.push_back(colapsar(trozo)); trozo.clear(); }
                res.push_back(colapsar(oracion.substr(0, c)));
                oracion = oracion.substr(c);
            }
            if (trozo.size() + oracion.size() > max_chars && !trozo.empty()) { res.push_back(colapsar(trozo)); trozo.clear(); }
            trozo += oracion;
        }
        if (colapsar(trozo).size() >= min_chars) res.push_back(colapsar(trozo));
    }
    // juntar trozos cortos consecutivos (párrafos breves de una página web) hasta max_chars:
    // menos encabezados por recuerdo y más contexto para que el modelo comprima mejor
    std::vector<std::string> juntos;
    for (auto & t : res) {
        if (!juntos.empty() && juntos.back().size() + 1 + t.size() <= max_chars) juntos.back() += " " + t;
        else juntos.push_back(t);
    }
    return juntos;
}

// ---------------------------------------------------------------------------
// rANS de 64 bits con probabilidades cuantizadas (deterministas) desde los logits
// ---------------------------------------------------------------------------
static const uint32_t ESCALA_BITS = 24;
static const uint32_t ESCALA      = 1u << ESCALA_BITS;
static const uint64_t RANS_L      = 1ull << 31;

static void cdf_desde_logits(const float * logits, int n_vocab, std::vector<uint32_t> & cum) {
    double lmax = logits[0];
    int    vmax = 0;
    for (int v = 1; v < n_vocab; ++v) if (logits[v] > lmax) { lmax = logits[v]; vmax = v; }
    static thread_local std::vector<double>   e;
    static thread_local std::vector<uint32_t> fr;
    e.resize(n_vocab);
    fr.resize(n_vocab);
    double z = 0.0;
    for (int v = 0; v < n_vocab; ++v) {
        const double l = logits[v];
        e[v] = (l == l) ? std::exp(l - lmax) : 0.0;
        z += e[v];
    }
    const double resto = (double) (ESCALA - (uint32_t) n_vocab);
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

// ---------------------------------------------------------------------------
// Motor
// ---------------------------------------------------------------------------
static const int MAX_TOKENS_RECUERDO = 250;
static const int N_CTX_MOTOR         = 1024;

motor::motor(llama_model * m, int n_threads, std::vector<region> rs) : model(m), regs(std::move(rs)) {
    if (regs.empty()) regs = regiones_por_defecto();
    if (regs.size() > 32) regs.resize(32);
    vocab   = llama_model_get_vocab(model);
    n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = N_CTX_MOTOR;
    cp.n_batch         = N_CTX_MOTOR;
    cp.n_ubatch        = 512;
    cp.n_seq_max       = 1;
    cp.n_threads       = std::max(1, n_threads);
    cp.n_threads_batch = std::max(1, n_threads);
    cp.no_perf         = true;
    ctx = llama_init_from_model(model, cp);

    // Huella: si cambia el modelo o las regiones, los códigos guardados no sirven.
    std::string h = std::to_string(n_vocab) + "/" + std::to_string(llama_model_n_embd(model)) + "/" +
                    std::to_string(llama_model_n_layer(model)) + "/" + std::to_string(llama_model_n_params(model));
    for (auto & r : regs) h += "|" + r.nombre;
    huella_ = ((uint64_t) fnv32(h) << 32) | fnv32(h + "#nai3");
}

motor::~motor() {
    if (ctx) llama_free(ctx);
}

std::vector<llama_token> motor::tokenizar(const std::string & s, bool especiales) const {
    std::vector<llama_token> t(s.size() + 16);
    int n = llama_tokenize(vocab, s.c_str(), (int) s.size(), t.data(), (int) t.size(), especiales, especiales);
    if (n < 0) {
        t.resize(-n);
        n = llama_tokenize(vocab, s.c_str(), (int) s.size(), t.data(), (int) t.size(), especiales, especiales);
    }
    t.resize(std::max(0, n));
    return t;
}

std::string motor::detokenizar(const std::vector<llama_token> & toks) const {
    std::string s;
    char buf[256];
    for (auto t : toks) {
        const int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, false);
        if (n > 0) s.append(buf, n);
    }
    return s;
}

bool motor::evaluar(const std::vector<llama_token> & t) {
    if (t.empty()) return true;
    std::vector<llama_token> copia = t;
    return llama_decode(ctx, llama_batch_get_one(copia.data(), (int) copia.size())) == 0;
}

std::string motor::plantilla(const std::string & usuario) const {
    if (const char * tmpl = llama_model_chat_template(model, nullptr)) {
        llama_chat_message msg = { "user", usuario.c_str() };
        std::vector<char> buf(usuario.size() * 2 + 1024);
        int n = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), (int) buf.size());
        if (n > (int) buf.size()) {
            buf.resize(n);
            n = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), (int) buf.size());
        }
        if (n > 0) return std::string(buf.data(), n);
    }
    return usuario + "\n";
}

std::string motor::prefijo(int region, int nivel, int rol) const {
    std::string p = "Tema: " + regs[std::min<size_t>(region, regs.size() - 1)].nombre + ".\n";
    if (rol == USUARIO)   p += "Usuario: ";
    if (rol == ASISTENTE) p += "Asistente: ";
    if (nivel == IDEA)    p += "Idea esencial: ";
    if (nivel == PISTA)   p += "Palabras clave: ";
    return p;
}

// Diccionario de tokens: cada ID en 7 bits por byte (bit alto = continúa), empacado en palabras de 32 bits.
std::vector<uint32_t> empacar_tokens(const std::vector<llama_token> & t) {
    std::vector<uint8_t> b;
    for (llama_token v : t) {
        uint32_t x = (uint32_t) v;
        while (x >= 128) { b.push_back((uint8_t) ((x & 127) | 128)); x >>= 7; }
        b.push_back((uint8_t) x);
    }
    std::vector<uint32_t> w((b.size() + 3) / 4, 0);
    for (size_t i = 0; i < b.size(); ++i) w[i / 4] |= (uint32_t) b[i] << (8 * (i % 4));
    return w;
}

bool desempacar_tokens(const codigo & c, int n_vocab, std::vector<llama_token> & t) {
    t.clear();
    const size_t nb = 4 * c.palabras.size();
    size_t i = 0;
    while ((int) t.size() < c.n_tokens) {
        uint32_t x = 0;
        int s = 0;
        for (;;) {
            if (i >= nb || s > 28) return false;
            const uint8_t byte = (uint8_t) (c.palabras[i / 4] >> (8 * (i % 4)));
            ++i;
            x |= (uint32_t) (byte & 127) << s;
            s += 7;
            if (!(byte & 128)) break;
        }
        if ((int) x >= n_vocab) return false;
        t.push_back((llama_token) x);
    }
    return true;
}

bool motor::codificar(const std::string & pre, const std::string & texto, codigo & out,
                      std::string * texto_tokens, std::vector<std::pair<std::string, double>> * palabras) {
    std::vector<llama_token> tp = tokenizar(pre, true);
    std::vector<llama_token> t  = tokenizar(texto, false);
    if (t.empty() || tp.empty()) return false;
    if ((int) t.size() > MAX_TOKENS_RECUERDO) t.resize(MAX_TOKENS_RECUERDO);
    if ((int) (tp.size() + t.size()) >= N_CTX_MOTOR) return false;

    // portable sin necesidad de medir sorpresa: ni siquiera se pasa por el modelo
    if (!con_modelo && !palabras) {
        out.tokens   = true;
        out.palabras = empacar_tokens(t);
        out.n_tokens = (uint8_t) t.size();
        if (texto_tokens) *texto_tokens = detokenizar(t);
        return out.palabras.size() <= 255;
    }

    llama_memory_clear(llama_get_memory(ctx), true);
    if (!evaluar(tp)) return false;

    std::vector<std::pair<uint32_t, uint32_t>> simbolos;
    std::vector<double> bits;
    for (size_t i = 0; i < t.size(); ++i) {
        cdf_desde_logits(llama_get_logits_ith(ctx, -1), n_vocab, cum);
        const uint32_t ini = cum[t[i]], fr = cum[t[i] + 1] - cum[t[i]];
        simbolos.push_back({ ini, fr });
        bits.push_back(-std::log2((double) fr / ESCALA));
        if (i + 1 < t.size() && !evaluar({ t[i] })) return false;
    }
    if (con_modelo) {
        rans_enc enc;
        for (size_t k = simbolos.size(); k-- > 0;) enc.poner(simbolos[k].first, simbolos[k].second);
        out.tokens   = false;
        out.palabras = enc.terminar();
    } else {
        out.tokens   = true;   // el modelo solo se usó para medir la sorpresa (claves y pistas)
        out.palabras = empacar_tokens(t);
    }
    out.n_tokens = (uint8_t) t.size();
    if (out.palabras.size() > 255) return false;

    const std::string tt = detokenizar(t);
    if (texto_tokens) *texto_tokens = tt;
    if (palabras) {
        // sorpresa por palabra: los bits de cada token se reparten entre sus bytes
        palabras->clear();
        std::string w;
        double bw = 0;
        char buf[256];
        for (size_t i = 0; i < t.size(); ++i) {
            const int n = llama_token_to_piece(vocab, t[i], buf, sizeof(buf), 0, false);
            for (int j = 0; j < n; ++j) {
                const unsigned char c = (unsigned char) buf[j];
                if (std::isalnum(c) || c >= 0x80) { w += (char) c; bw += bits[i] / n; }
                else if (!w.empty()) { palabras->push_back({ w, bw }); w.clear(); bw = 0; }
            }
        }
        if (!w.empty()) palabras->push_back({ w, bw });
    }
    return true;
}

bool motor::decodificar(const std::string & pre, const codigo & c, std::string & out) {
    out.clear();
    if (c.tokens) { // diccionario de tokens: consulta de tabla, sin modelo
        std::vector<llama_token> t;
        if (c.n_tokens == 0 || !desempacar_tokens(c, n_vocab, t)) return false;
        out = detokenizar(t);
        return true;
    }
    if (c.n_tokens == 0 || c.palabras.size() < 2) return false;
    std::vector<llama_token> tp = tokenizar(pre, true);
    llama_memory_clear(llama_get_memory(ctx), true);
    if (!evaluar(tp)) return false;
    rans_dec dec;
    dec.iniciar(c.palabras);
    std::vector<llama_token> t;
    for (int i = 0; i < c.n_tokens; ++i) {
        cdf_desde_logits(llama_get_logits_ith(ctx, -1), n_vocab, cum);
        const uint32_t m = dec.mirar();
        const int v = (int) (std::upper_bound(cum.begin(), cum.end(), m) - cum.begin()) - 1;
        if (v < 0 || v >= n_vocab) return false;
        dec.avanzar(cum[v], cum[v + 1] - cum[v]);
        t.push_back(v);
        if (i + 1 < c.n_tokens && !evaluar({ v })) return false;
    }
    out = detokenizar(t);
    return true;
}

int motor::clasificar(const std::string & texto) {
    if (regs.size() <= 1 || !ctx) return 0;
    std::string lista, gramatica = "root ::= ";
    for (size_t i = 0; i < regs.size(); ++i) {
        lista += (i ? ", " : "") + regs[i].nombre;
        std::string esc;
        for (char c : regs[i].nombre) { if (c == '"' || c == '\\') esc += '\\'; esc += c; }
        gramatica += (i ? " | \"" : "\"") + esc + "\"";
    }
    const std::string prompt = plantilla(
        "Elige el tema que mejor describe el texto.\nTemas: " + lista + ".\nTexto: «" + recortar(colapsar(texto), 700) +
        "»\nResponde solo con el nombre del tema.");
    std::vector<llama_token> tp = tokenizar(prompt, true);
    if ((int) tp.size() >= N_CTX_MOTOR - 16) tp.resize(N_CTX_MOTOR - 16);
    llama_memory_clear(llama_get_memory(ctx), true);
    if (!evaluar(tp)) return (int) regs.size() - 1;

    llama_sampler * s = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler * g = llama_sampler_init_grammar(vocab, gramatica.c_str(), "root");
    if (!g) { llama_sampler_free(s); return (int) regs.size() - 1; }
    llama_sampler_chain_add(s, g);
    llama_sampler_chain_add(s, llama_sampler_init_greedy());
    std::vector<llama_token> salida;
    for (int i = 0; i < 16; ++i) {
        const llama_token t = llama_sampler_sample(s, ctx, -1);
        if (llama_vocab_is_eog(vocab, t)) break;
        salida.push_back(t);
        if (!evaluar({ t })) break;
    }
    llama_sampler_free(s);
    const std::string r = detokenizar(salida);
    int mejor = -1;
    size_t largo = 0;
    for (size_t i = 0; i < regs.size(); ++i) {
        if (r == regs[i].nombre) return (int) i;
        if (r.size() > largo && regs[i].nombre.rfind(r, 0) == 0) { mejor = (int) i; largo = r.size(); }
    }
    return mejor >= 0 ? mejor : (int) regs.size() - 1;
}

std::string motor::generar(const std::string & instruccion, int n_max) {
    if (!ctx) return "";
    std::vector<llama_token> tp = tokenizar(plantilla(instruccion), true);
    if ((int) tp.size() + n_max >= N_CTX_MOTOR) tp.resize(std::max(0, N_CTX_MOTOR - n_max - 1));
    llama_memory_clear(llama_get_memory(ctx), true);
    if (!evaluar(tp)) return "";
    std::vector<llama_token> out;
    for (int i = 0; i < n_max; ++i) {
        const float * l = llama_get_logits_ith(ctx, -1);
        llama_token t = 0;
        for (int v = 1; v < n_vocab; ++v) if (l[v] > l[t]) t = v;
        if (llama_vocab_is_eog(vocab, t)) break;
        const std::string pieza = detokenizar({ t });
        if (pieza.find('\n') != std::string::npos && !out.empty()) break;
        out.push_back(t);
        if (!evaluar({ t })) break;
    }
    std::string s = colapsar(detokenizar(out));
    // quitar comillas o guiones que el modelo agregue alrededor
    while (!s.empty() && (s.front() == '"' || s.front() == '-' || s.front() == ' ')) s.erase(0, 1);
    while (!s.empty() && (s.back() == '"' || s.back() == ' ')) s.pop_back();
    return s;
}

static std::vector<uint16_t> mejores_claves(const std::vector<std::pair<std::string, double>> & pals, size_t n,
                                            std::vector<std::string> * texto = nullptr) {
    std::vector<std::pair<double, std::string>> orden;
    for (auto & p : pals) if (clave_palabra(p.first)) orden.push_back({ p.second, p.first });
    std::stable_sort(orden.begin(), orden.end(), [](auto & a, auto & b) { return a.first > b.first; });
    std::vector<uint16_t> res;
    for (auto & o : orden) {
        const uint16_t k = clave_palabra(o.second);
        if (std::find(res.begin(), res.end(), k) != res.end()) continue;
        res.push_back(k);
        if (texto) texto->push_back(o.second);
        if (res.size() == n) break;
    }
    return res;
}

int motor::agregar(memoria & m, const std::string & texto, int region, int rol) {
    if (!ctx) return 0;
    int n = 0;
    for (auto & trozo : trocear(texto)) {
        const int reg = region >= 0 && region < (int) regs.size() ? region : clasificar(trozo);
        recuerdo r;
        r.region = (uint8_t) reg;
        r.rol    = (uint8_t) rol;
        r.nivel  = EXACTO;
        r.dia    = hoy();
        std::string tt;
        std::vector<std::pair<std::string, double>> pals;
        if (!codificar(prefijo(reg, EXACTO, rol), trozo, r.cod, &tt, &pals)) continue;
        r.claves = mejores_claves(pals, rol == DOCUMENTO ? 6 : 4);
        r.check  = check8(tt);
        r.cache  = tt;
        m.recuerdos.push_back(std::move(r));
        ++n;
    }
    if (n) { m.sucia = true; ajustar(m); }
    return n;
}

const std::string & motor::texto(recuerdo & r) {
    if (!r.cache.empty() || r.danado || r.cod.n_tokens == 0) return r.cache;
    std::string s;
    if (decodificar(prefijo(r.region, r.nivel, r.rol), r.cod, s) && check8(s) == r.check) {
        r.cache = s;
    } else {
        r.danado = true; // modelo o backend distinto al que lo codificó
    }
    return r.cache;
}

double motor::importancia(const recuerdo & r) const {
    const int peso = regs[std::min<size_t>(r.region, regs.size() - 1)].peso;
    const int dias = std::max(0, (int) hoy() - (int) r.dia);
    return peso * (1.0 + 0.5 * r.usos) + 1.0 / (1.0 + dias);
}

void motor::bajar_nivel(recuerdo & r) {
    const std::string txt = texto(r);
    if (r.danado || txt.empty()) { r.nivel = PISTA; r.cod = codigo(); r.cache.clear(); return; }

    if (r.nivel == EXACTO) {
        const std::string idea = generar(
            "Resume en una sola frase corta, de máximo 12 palabras, la idea esencial de este texto. "
            "Conserva nombres, cifras y fechas. Escribe solo la frase.\nTexto: «" + txt + "»", 32);
        codigo ci;
        std::string ti;
        if (!idea.empty() && codificar(prefijo(r.region, IDEA, r.rol), idea, ci, &ti) && ci.bytes() < r.cod.bytes()) {
            r.cod = ci; r.nivel = IDEA; r.cache = ti; r.check = check8(ti);
            return;
        }
    }
    // PISTA: las palabras más sorprendentes del texto actual
    codigo tmp;
    std::vector<std::pair<std::string, double>> pals;
    std::vector<std::string> ws;
    if (codificar(prefijo(r.region, EXACTO, r.rol), txt, tmp, nullptr, &pals)) mejores_claves(pals, 3, &ws);
    std::string pista;
    for (auto & w : ws) pista += (pista.empty() ? "" : ", ") + w;
    codigo cp;
    std::string tp;
    if (!pista.empty() && codificar(prefijo(r.region, PISTA, r.rol), pista, cp, &tp) && cp.bytes() < r.cod.bytes()) {
        r.cod = cp; r.cache = tp; r.check = check8(tp);
    } else {
        r.cod = codigo(); r.cache.clear();  // solo quedan las claves para volver a llegar
    }
    r.nivel = PISTA;
}

void motor::ajustar(memoria & m) {
    for (int guardia = 0; m.bytes() > m.presupuesto && !m.recuerdos.empty() && guardia < 100000; ++guardia) {
        int peor = -1;
        double vp = 1e30;
        for (size_t i = 0; i < m.recuerdos.size(); ++i) {
            if (m.recuerdos[i].nivel >= PISTA) continue;
            const double v = importancia(m.recuerdos[i]);
            if (v < vp) { vp = v; peor = (int) i; }
        }
        if (peor >= 0) { bajar_nivel(m.recuerdos[peor]); m.sucia = true; continue; }
        // todo está en pista: se olvida por completo lo menos importante
        for (size_t i = 0; i < m.recuerdos.size(); ++i) {
            const double v = importancia(m.recuerdos[i]);
            if (v < vp) { vp = v; peor = (int) i; }
        }
        m.recuerdos.erase(m.recuerdos.begin() + peor);
        m.sucia = true;
    }
}

std::string motor::recordar(std::vector<memoria *> ms, const std::string & consulta, int max_rec, size_t max_chars, int * n_usados) {
    if (n_usados) *n_usados = 0;
    const std::vector<uint16_t> qk = claves_de(consulta);
    const std::set<uint16_t> qs(qk.begin(), qk.end());

    struct cand { memoria * m; size_t i; double score; int hits; bool asociado = false; };
    std::vector<cand> cs;
    size_t total = 0;
    for (auto * m : ms) {
        total += m->recuerdos.size();
        for (size_t i = 0; i < m->recuerdos.size(); ++i) {
            int hits = 0;
            for (auto k : m->recuerdos[i].claves) hits += qs.count(k);
            if (hits > 0) cs.push_back({ m, i, 10.0 * hits + importancia(m->recuerdos[i]), hits });
        }
    }
    // Sin coincidencias: el GPS va a la región de la consulta.
    if (cs.empty() && total > 0 && palabras_de(consulta).size() >= 3) {
        const int reg = clasificar(consulta);
        for (auto * m : ms) {
            for (size_t i = 0; i < m->recuerdos.size(); ++i) {
                if (m->recuerdos[i].region == reg) cs.push_back({ m, i, importancia(m->recuerdos[i]), 0 });
            }
        }
        std::sort(cs.begin(), cs.end(), [](auto & a, auto & b) { return a.score > b.score; });
        if (cs.size() > 2) cs.resize(2);
    }
    // Si hay recuerdos con varias coincidencias, los de una sola (posible colisión del hash
    // de 16 bits o palabra suelta) quedan fuera.
    int max_hits = 0;
    for (auto & c : cs) max_hits = std::max(max_hits, c.hits);
    if (max_hits >= 2) cs.erase(std::remove_if(cs.begin(), cs.end(), [&](auto & c) { return c.hits < max_hits - 1; }), cs.end());
    std::stable_sort(cs.begin(), cs.end(), [](auto & a, auto & b) { return a.score > b.score; });

    // Segundo salto (activación propagada): las claves de los mejores recuerdos directos que la
    // consulta no tenía activan otros recuerdos. Exige 2+ claves compartidas (una sola puede ser
    // colisión del hash) y pesa menos que una coincidencia directa: es intuición, no evidencia.
    if (max_hits > 0) {
        std::set<uint16_t> q2;
        for (size_t k = 0; k < cs.size() && k < 3; ++k) {
            for (auto c : cs[k].m->recuerdos[cs[k].i].claves) if (!qs.count(c)) q2.insert(c);
        }
        std::set<std::pair<memoria *, size_t>> ya;
        for (auto & c : cs) ya.insert({ c.m, c.i });
        std::vector<cand> sal;
        for (auto * m : ms) {
            for (size_t i = 0; i < m->recuerdos.size() && !q2.empty(); ++i) {
                if (ya.count({ m, i })) continue;
                int h = 0;
                for (auto k : m->recuerdos[i].claves) h += q2.count(k);
                if (h >= 2) sal.push_back({ m, i, 4.0 * h + importancia(m->recuerdos[i]), 0, true });
            }
        }
        std::stable_sort(sal.begin(), sal.end(), [](auto & a, auto & b) { return a.score > b.score; });
        if (sal.size() > 2) sal.resize(2);
        cs.insert(cs.end(), sal.begin(), sal.end());
    }

    std::map<int, int> por_region;
    std::string lineas;
    int n = 0;
    for (auto & c : cs) {
        if (n >= max_rec) break;
        recuerdo & r = c.m->recuerdos[c.i];
        if (por_region[r.region] >= 2) continue; // diversidad: cruza regiones
        const std::string & t = texto(r);
        if (t.empty()) continue;
        std::string l = "- (" + regs[std::min<size_t>(r.region, regs.size() - 1)].nombre + ", " + nombre_rol(r.rol);
        if (r.nivel != EXACTO) l += std::string(", ") + nombre_nivel(r.nivel);
        if (c.asociado)        l += ", asociado";
        l += ") " + t + "\n";
        if (lineas.size() + l.size() > max_chars) break;
        lineas += l;
        por_region[r.region]++;
        r.usos = (uint8_t) std::min(255, r.usos + 1);
        r.dia  = hoy();
        c.m->sucia = true;
        ++n;
    }
    if (n_usados) *n_usados = n;
    if (lineas.empty()) return "";
    return "Recuerdos relevantes de conversaciones y documentos anteriores (memoria NAI). "
           "Úsalos solo si ayudan a responder:\n" + lineas;
}

size_t memoria::bytes() const {
    size_t b = 0;
    for (auto & r : recuerdos) b += r.bytes();
    return b;
}

estadisticas motor::medir(const memoria & m) const {
    estadisticas e;
    e.presupuesto = m.presupuesto;
    e.bytes = m.bytes();
    for (auto & r : m.recuerdos) {
        if (r.nivel == EXACTO) { e.n_exacto++; e.bytes_texto += r.cache.size(); }
        if (r.nivel == IDEA)   e.n_idea++;
        if (r.nivel == PISTA)  e.n_pista++;
        if (r.danado)          e.n_danados++;
    }
    return e;
}

// ---------------------------------------------------------------------------
// Archivo: "NAIM" v3 | huella | memoria de trabajo | recuerdos
// ---------------------------------------------------------------------------
static void w8 (std::string & b, uint8_t v)  { b += (char) v; }
static void w16(std::string & b, uint16_t v) { w8(b, v & 0xFF); w8(b, v >> 8); }
static void w32(std::string & b, uint32_t v) { w16(b, v & 0xFFFF); w16(b, v >> 16); }
static void w64(std::string & b, uint64_t v) { w32(b, (uint32_t) v); w32(b, (uint32_t) (v >> 32)); }

struct lector {
    const std::string & b;
    size_t i = 0;
    bool ok = true;
    uint8_t r8() { if (i + 1 > b.size()) { ok = false; return 0; } return (uint8_t) b[i++]; }
    uint16_t r16() { uint16_t a = r8(); return a | (uint16_t) (r8() << 8); }
    uint32_t r32() { uint32_t a = r16(); return a | ((uint32_t) r16() << 16); }
    uint64_t r64() { uint64_t a = r32(); return a | ((uint64_t) r32() << 32); }
    std::string rs(size_t n) { if (i + n > b.size()) { ok = false; return ""; } std::string s = b.substr(i, n); i += n; return s; }
};

bool motor::guardar(const memoria & m, const std::string & ruta) const {
    std::string b = "NAIM";
    w8(b, 3);
    w64(b, huella_);
    const std::string pu = recortar(m.previo_usuario, 4000), pa = recortar(m.previo_asistente, 4000);
    w16(b, (uint16_t) pu.size()); b += pu;
    w16(b, (uint16_t) pa.size()); b += pa;
    w32(b, (uint32_t) m.recuerdos.size());
    for (auto & r : m.recuerdos) {
        w8(b, r.region);
        w8(b, (uint8_t) ((r.nivel & 3) | ((r.rol & 3) << 2) | (r.danado ? 16 : 0) | (r.cod.tokens ? 32 : 0)));
        w8(b, r.usos);
        w8(b, r.check);
        w16(b, r.dia);
        w8(b, (uint8_t) std::min<size_t>(r.claves.size(), 255));
        for (size_t k = 0; k < r.claves.size() && k < 255; ++k) w16(b, r.claves[k]);
        w8(b, r.cod.n_tokens);
        w8(b, (uint8_t) r.cod.palabras.size());
        for (auto p : r.cod.palabras) w32(b, p);
    }
    if (cla.activa) {
        b = nai_cifrado::cifrar(cla, b, std::filesystem::path(ruta).filename().string());
    }
    const std::string tmp = ruta + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        f.write(b.data(), (std::streamsize) b.size());
        if (!f) return false;
    }
    std::error_code ec;
    std::filesystem::permissions(tmp, std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
                                 std::filesystem::perm_options::replace, ec);
    std::filesystem::rename(tmp, ruta, ec);
    if (ec) { std::filesystem::remove(tmp, ec); return false; }
    return true;
}

bool motor::cargar(memoria & m, const std::string & ruta) const {
    std::ifstream f(ruta, std::ios::binary);
    if (!f) return false;
    std::string b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (nai_cifrado::es_cifrado(b)) {
        std::string plano;
        if (!cla.activa || !nai_cifrado::descifrar(cla, b, std::filesystem::path(ruta).filename().string(), plano)) {
            fprintf(stderr, "[NAI] memoria: no se pudo descifrar %s (falta la clave, es otra clave o el archivo fue alterado)\n", ruta.c_str());
            return false;
        }
        b = std::move(plano);
    }
    lector l{ b };
    if (l.rs(4) != "NAIM" || l.r8() != 3 || l.r64() != huella_ || !l.ok) return false;
    memoria n;
    n.presupuesto = m.presupuesto;
    n.previo_usuario   = l.rs(l.r16());
    n.previo_asistente = l.rs(l.r16());
    const uint32_t cnt = l.r32();
    for (uint32_t k = 0; k < cnt && l.ok; ++k) {
        recuerdo r;
        r.region = l.r8();
        const uint8_t nr = l.r8();
        r.nivel  = nr & 3;
        r.rol    = (nr >> 2) & 3;
        r.danado = (nr & 16) != 0;
        r.cod.tokens = (nr & 32) != 0;   // diccionario de tokens (portable)
        r.usos   = l.r8();
        r.check  = l.r8();
        r.dia    = l.r16();
        const uint8_t nk = l.r8();
        for (int j = 0; j < nk; ++j) r.claves.push_back(l.r16());
        r.cod.n_tokens = l.r8();
        const uint8_t np = l.r8();
        for (int j = 0; j < np; ++j) r.cod.palabras.push_back(l.r32());
        if (r.region >= regs.size()) r.region = (uint8_t) (regs.size() - 1);
        if (l.ok) n.recuerdos.push_back(std::move(r));
    }
    if (!l.ok) return false;
    m = std::move(n);
    return true;
}

// ---------------------------------------------------------------------------
// Servicio
// ---------------------------------------------------------------------------
static int64_t ahora_seg() { return (int64_t) std::time(nullptr); }

servicio::servicio(llama_model * model, const config & c) : cfg(c) {
    std::vector<region> regs = cfg.ruta_regiones.empty() ? regiones_por_defecto() : leer_regiones(cfg.ruta_regiones);
    mot = std::make_unique<motor>(model, cfg.n_threads, regs);
    mot->usar_clave(cfg.clave);
    mot->usar_codificacion_modelo(cfg.compacta);
    std::error_code ec;
    std::filesystem::create_directories(cfg.directorio, ec);
    biblioteca.presupuesto = cfg.bytes_biblioteca;
    if (mot->ok()) mot->cargar(biblioteca, (std::filesystem::path(cfg.directorio) / "biblioteca.mem").string());
    hilo = std::thread([this] { bucle(); });
}

servicio::~servicio() {
    {
        std::lock_guard<std::mutex> lk(mtx_cola);
        salir = true;
    }
    cv.notify_all();
    if (hilo.joinable()) hilo.join();
    std::lock_guard<std::mutex> lk(mtx);
    for (auto & kv : sesiones) if (kv.second.sucia) mot->guardar(kv.second, ruta_sesion(kv.first));
    if (biblioteca.sucia) mot->guardar(biblioteca, (std::filesystem::path(cfg.directorio) / "biblioteca.mem").string());
}

std::string servicio::ruta_sesion(uint64_t s) const {
    char nombre[64];
    snprintf(nombre, sizeof(nombre), "nai-%016llx.mem", (unsigned long long) s);
    return (std::filesystem::path(cfg.directorio) / nombre).string();
}

memoria & servicio::sesion_mem(uint64_t s) {
    auto it = sesiones.find(s);
    if (it == sesiones.end()) {
        memoria m;
        m.presupuesto = cfg.bytes_sesion;
        mot->cargar(m, ruta_sesion(s));
        m.presupuesto = cfg.bytes_sesion;
        it = sesiones.emplace(s, std::move(m)).first;
    }
    it->second.t_acceso = ahora_seg();
    return it->second;
}

void servicio::purgar() {
    const int64_t t = ahora_seg();
    if (t - t_purga < 60) return;
    t_purga = t;
    // en RAM: lo que caducó se olvida; lo inactivo más de 10 min se guarda y se descarga
    for (auto it = sesiones.begin(); it != sesiones.end();) {
        const int64_t inactivo = t - it->second.t_acceso;
        if (inactivo >= cfg.ttl_seg) { it = sesiones.erase(it); continue; }
        if (inactivo >= 600) {
            if (it->second.sucia) mot->guardar(it->second, ruta_sesion(it->first));
            it = sesiones.erase(it);
            continue;
        }
        ++it;
    }
    // en disco: archivos de sesión sin actividad durante el TTL
    std::error_code ec;
    for (auto & e : std::filesystem::directory_iterator(cfg.directorio, ec)) {
        const std::string n = e.path().filename().string();
        if (n.rfind("nai-", 0) != 0 || e.path().extension() != ".mem") continue;
        const auto ft  = std::filesystem::last_write_time(e.path(), ec);
        const auto sys = std::chrono::time_point_cast<std::chrono::seconds>(
            ft - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());
        if (t - (int64_t) sys.time_since_epoch().count() >= cfg.ttl_seg) std::filesystem::remove(e.path(), ec);
    }
}

contexto_recordado servicio::preparar(uint64_t s, const std::string & consulta) {
    contexto_recordado res;
    if (!ok()) return res;
    std::lock_guard<std::mutex> lk(mtx);
    purgar();
    memoria & m = sesion_mem(s);
    const auto us = partir_turnos(m.previo_usuario), as = partir_turnos(m.previo_asistente);
    if (us.size() == as.size()) {
        for (size_t i = 0; i < us.size(); ++i) {
            if (!us[i].empty() && !as[i].empty()) res.turnos.push_back({ us[i], as[i] });
        }
    }
    res.bloque = mot->recordar({ &m, &biblioteca }, consulta, cfg.max_recuerdos, cfg.max_chars, &res.n_recuerdos);
    return res;
}

void servicio::registrar_turno(uint64_t s, const std::string & usuario, const std::string & asistente) {
    if (!ok()) return;
    encolar([this, s, usuario, asistente] {
        std::lock_guard<std::mutex> lk(mtx);
        memoria & m = sesion_mem(s);
        const int reg = mot->clasificar(usuario + "\n" + recortar(asistente, 400));
        mot->agregar(m, usuario, reg, USUARIO);
        mot->agregar(m, asistente, reg, ASISTENTE);
        anillo_turnos(m, usuario, asistente, cfg.turnos, 3600);   // 3600 < 4000 que admite el archivo
        if (mot->guardar(m, ruta_sesion(s))) m.sucia = false;
        else fprintf(stderr, "%s\n", "[NAI] memoria: no se pudo guardar la sesión");
    });
}

int servicio::indice_region(const std::string & nombre) const {
    if (!mot) return -1;
    const auto & rs = mot->regiones();
    for (size_t i = 0; i < rs.size(); ++i) if (rs[i].nombre == nombre) return (int) i;
    return -1;
}

int servicio::aprender(const std::string & texto, const std::string & fuente, const std::string & region) {
    if (!ok()) return 0;
    const int reg = region.empty() ? -1 : indice_region(region);
    std::vector<std::string> trozos = trocear(texto);
    {
        std::lock_guard<std::mutex> lk(mtx_cola);
        pendientes_aprender += trozos.size();
    }
    const std::string ruta = (std::filesystem::path(cfg.directorio) / "biblioteca.mem").string();
    for (size_t i = 0; i < trozos.size(); ++i) {
        const bool ultimo = i + 1 == trozos.size();
        encolar([this, t = trozos[i], ultimo, ruta, reg] {
            std::lock_guard<std::mutex> lk(mtx);
            mot->agregar(biblioteca, t, reg, DOCUMENTO);
            {
                std::lock_guard<std::mutex> lc(mtx_cola);
                pendientes_aprender--;
                aprendidos++;
            }
            if ((ultimo || aprendidos % 16 == 0) && mot->guardar(biblioteca, ruta)) biblioteca.sucia = false;
        });
    }
    (void) fuente;
    return (int) trozos.size();
}

std::string servicio::estado_json() {
    estadisticas e;
    size_t n_ses = 0;
    {
        std::lock_guard<std::mutex> lk(mtx);
        e = mot->medir(biblioteca);
        n_ses = sesiones.size();
    }
    size_t pend, apr;
    {
        std::lock_guard<std::mutex> lc(mtx_cola);
        pend = pendientes_aprender;
        apr  = aprendidos;
    }
    char buf[512];
    snprintf(buf, sizeof(buf),
             "{\"biblioteca\":{\"recuerdos\":%zu,\"exacto\":%zu,\"idea\":%zu,\"pista\":%zu,\"danados\":%zu,"
             "\"bytes\":%zu,\"presupuesto\":%zu},\"sesiones_en_ram\":%zu,\"trozos_pendientes\":%zu,"
             "\"trozos_aprendidos\":%zu,\"bytes_por_sesion\":%zu,\"ttl_seg\":%lld}",
             e.n_exacto + e.n_idea + e.n_pista, e.n_exacto, e.n_idea, e.n_pista, e.n_danados, e.bytes,
             e.presupuesto, n_ses, pend, apr, cfg.bytes_sesion, (long long) cfg.ttl_seg);
    return buf;
}

void servicio::encolar(std::function<void()> f) {
    {
        std::lock_guard<std::mutex> lk(mtx_cola);
        cola.push_back(std::move(f));
    }
    cv.notify_one();
}

void servicio::bucle() {
    for (;;) {
        std::function<void()> f;
        {
            std::unique_lock<std::mutex> lk(mtx_cola);
            cv.wait(lk, [this] { return salir || !cola.empty(); });
            if (salir && cola.empty()) return;
            f = std::move(cola.front());
            cola.pop_front();
        }
        // cede el paso a las peticiones en curso (máx. 10 s, luego avanza para no atorar la cola)
        for (int i = 0; i < 1000 && activas.load() > 0 && !salir; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        try { f(); }
        catch (const std::exception & e) { fprintf(stderr, "[NAI] memoria: error en segundo plano: %s\n", e.what()); }
        catch (...) { fprintf(stderr, "%s\n", "[NAI] memoria: error desconocido en segundo plano"); }
    }
}

} // namespace nai_mem
