// [NAI] Cifrado autenticado de sesiones y memorias: ChaCha20-Poly1305 (RFC 8439).
//
// Los archivos de sesión (KV) y de memoria (.mem) guardan conversaciones; con --nai-cifrado
// se cifran en disco. ChaCha20-Poly1305 cifra Y autentica: si alguien altera el archivo, o lo
// copia con el nombre de otra sesión (el nombre va en los datos asociados), no se descifra.
//
// Formato:  "NAIC" | versión (1) | nonce (12) | texto cifrado | etiqueta (16)
//
// La clave (32 bytes, en hexadecimal) vive en un archivo aparte que debe quedar FUERA de la
// carpeta de sesiones: si se copian juntos, el cifrado no protege nada. Sin la clave, los
// archivos cifrados no se pueden recuperar.
//
// Implementación portable (sin dependencias) verificada con los vectores del RFC 8439 en
// tests/test-nai.cpp.
#pragma once

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>

namespace nai_cifrado {

struct clave {
    uint8_t k[32] = {};
    bool    activa = false;
};

namespace detalle {

inline uint32_t rotl(uint32_t v, int c) { return (v << c) | (v >> (32 - c)); }
inline uint32_t le32(const uint8_t * p) { return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24); }
inline void pon32(uint8_t * p, uint32_t v) { p[0] = (uint8_t) v; p[1] = (uint8_t) (v >> 8); p[2] = (uint8_t) (v >> 16); p[3] = (uint8_t) (v >> 24); }

inline void cuarto(uint32_t & a, uint32_t & b, uint32_t & c, uint32_t & d) {
    a += b; d ^= a; d = rotl(d, 16);
    c += d; b ^= c; b = rotl(b, 12);
    a += b; d ^= a; d = rotl(d, 8);
    c += d; b ^= c; b = rotl(b, 7);
}

// Bloque ChaCha20 (RFC 8439 §2.3)
inline void bloque(const uint8_t key[32], uint32_t contador, const uint8_t nonce[12], uint8_t out[64]) {
    uint32_t s[16] = { 0x61707865, 0x3320646e, 0x79622d32, 0x6b206574 };
    for (int i = 0; i < 8; ++i) s[4 + i] = le32(key + 4 * i);
    s[12] = contador;
    for (int i = 0; i < 3; ++i) s[13 + i] = le32(nonce + 4 * i);
    uint32_t x[16];
    memcpy(x, s, sizeof(s));
    for (int i = 0; i < 10; ++i) {
        cuarto(x[0], x[4], x[8], x[12]); cuarto(x[1], x[5], x[9], x[13]);
        cuarto(x[2], x[6], x[10], x[14]); cuarto(x[3], x[7], x[11], x[15]);
        cuarto(x[0], x[5], x[10], x[15]); cuarto(x[1], x[6], x[11], x[12]);
        cuarto(x[2], x[7], x[8], x[13]); cuarto(x[3], x[4], x[9], x[14]);
    }
    for (int i = 0; i < 16; ++i) pon32(out + 4 * i, x[i] + s[i]);
}

} // namespace detalle

// ChaCha20 (RFC 8439 §2.4): XOR del flujo sobre datos, empezando en `contador`.
inline void chacha20(const uint8_t key[32], uint32_t contador, const uint8_t nonce[12], uint8_t * datos, size_t n) {
    uint8_t ks[64];
    for (size_t i = 0; i < n; i += 64, ++contador) {
        detalle::bloque(key, contador, nonce, ks);
        const size_t m = n - i < 64 ? n - i : 64;
        for (size_t j = 0; j < m; ++j) datos[i + j] ^= ks[j];
    }
}

// Poly1305 (RFC 8439 §2.5), aritmética de 26 bits (portable, estilo poly1305-donna).
class poly1305 {
public:
    explicit poly1305(const uint8_t key[32]) {
        using detalle::le32;
        r[0] = (le32(key + 0)) & 0x3ffffff;
        r[1] = (le32(key + 3) >> 2) & 0x3ffff03;
        r[2] = (le32(key + 6) >> 4) & 0x3ffc0ff;
        r[3] = (le32(key + 9) >> 6) & 0x3f03fff;
        r[4] = (le32(key + 12) >> 8) & 0x00fffff;
        for (int i = 0; i < 4; ++i) pad[i] = le32(key + 16 + 4 * i);
    }
    void actualizar(const uint8_t * m, size_t n) {
        if (lleno) {
            const size_t t = 16 - lleno < n ? 16 - lleno : n;
            memcpy(buf + lleno, m, t);
            lleno += t; m += t; n -= t;
            if (lleno < 16) return;
            bloques(buf, 16, false);
            lleno = 0;
        }
        const size_t completos = n & ~(size_t) 15;
        if (completos) { bloques(m, completos, false); m += completos; n -= completos; }
        if (n) { memcpy(buf, m, n); lleno = n; }
    }
    void terminar(uint8_t tag[16]) {
        if (lleno) {
            buf[lleno] = 1;
            for (size_t i = lleno + 1; i < 16; ++i) buf[i] = 0;
            bloques(buf, 16, true);
        }
        uint32_t h0 = h[0], h1 = h[1], h2 = h[2], h3 = h[3], h4 = h[4], c;
        c = h1 >> 26; h1 &= 0x3ffffff; h2 += c;
        c = h2 >> 26; h2 &= 0x3ffffff; h3 += c;
        c = h3 >> 26; h3 &= 0x3ffffff; h4 += c;
        c = h4 >> 26; h4 &= 0x3ffffff; h0 += c * 5;
        c = h0 >> 26; h0 &= 0x3ffffff; h1 += c;
        // h - p
        uint32_t g0 = h0 + 5; c = g0 >> 26; g0 &= 0x3ffffff;
        uint32_t g1 = h1 + c; c = g1 >> 26; g1 &= 0x3ffffff;
        uint32_t g2 = h2 + c; c = g2 >> 26; g2 &= 0x3ffffff;
        uint32_t g3 = h3 + c; c = g3 >> 26; g3 &= 0x3ffffff;
        uint32_t g4 = h4 + c - (1u << 26);
        uint32_t mask = (g4 >> 31) - 1;  // todo unos si h >= p
        g0 &= mask; g1 &= mask; g2 &= mask; g3 &= mask; g4 &= mask;
        mask = ~mask;
        h0 = (h0 & mask) | g0; h1 = (h1 & mask) | g1; h2 = (h2 & mask) | g2; h3 = (h3 & mask) | g3; h4 = (h4 & mask) | g4;
        // h = h % 2^128
        h0 = (h0) | (h1 << 26);
        h1 = (h1 >> 6) | (h2 << 20);
        h2 = (h2 >> 12) | (h3 << 14);
        h3 = (h3 >> 18) | (h4 << 8);
        // tag = (h + pad) % 2^128
        uint64_t f;
        f = (uint64_t) h0 + pad[0];             h0 = (uint32_t) f;
        f = (uint64_t) h1 + pad[1] + (f >> 32); h1 = (uint32_t) f;
        f = (uint64_t) h2 + pad[2] + (f >> 32); h2 = (uint32_t) f;
        f = (uint64_t) h3 + pad[3] + (f >> 32); h3 = (uint32_t) f;
        detalle::pon32(tag + 0, h0); detalle::pon32(tag + 4, h1); detalle::pon32(tag + 8, h2); detalle::pon32(tag + 12, h3);
    }
private:
    void bloques(const uint8_t * m, size_t n, bool final) {
        using detalle::le32;
        const uint32_t hibit = final ? 0 : (1u << 24);
        const uint32_t r0 = r[0], r1 = r[1], r2 = r[2], r3 = r[3], r4 = r[4];
        const uint32_t s1 = r1 * 5, s2 = r2 * 5, s3 = r3 * 5, s4 = r4 * 5;
        uint32_t h0 = h[0], h1 = h[1], h2 = h[2], h3 = h[3], h4 = h[4];
        for (; n >= 16; m += 16, n -= 16) {
            h0 += (le32(m + 0)) & 0x3ffffff;
            h1 += (le32(m + 3) >> 2) & 0x3ffffff;
            h2 += (le32(m + 6) >> 4) & 0x3ffffff;
            h3 += (le32(m + 9) >> 6) & 0x3ffffff;
            h4 += (le32(m + 12) >> 8) | hibit;
            const uint64_t d0 = (uint64_t) h0 * r0 + (uint64_t) h1 * s4 + (uint64_t) h2 * s3 + (uint64_t) h3 * s2 + (uint64_t) h4 * s1;
            uint64_t       d1 = (uint64_t) h0 * r1 + (uint64_t) h1 * r0 + (uint64_t) h2 * s4 + (uint64_t) h3 * s3 + (uint64_t) h4 * s2;
            uint64_t       d2 = (uint64_t) h0 * r2 + (uint64_t) h1 * r1 + (uint64_t) h2 * r0 + (uint64_t) h3 * s4 + (uint64_t) h4 * s3;
            uint64_t       d3 = (uint64_t) h0 * r3 + (uint64_t) h1 * r2 + (uint64_t) h2 * r1 + (uint64_t) h3 * r0 + (uint64_t) h4 * s4;
            uint64_t       d4 = (uint64_t) h0 * r4 + (uint64_t) h1 * r3 + (uint64_t) h2 * r2 + (uint64_t) h3 * r1 + (uint64_t) h4 * r0;
            uint32_t c;
            c = (uint32_t) (d0 >> 26); h0 = (uint32_t) d0 & 0x3ffffff;
            d1 += c; c = (uint32_t) (d1 >> 26); h1 = (uint32_t) d1 & 0x3ffffff;
            d2 += c; c = (uint32_t) (d2 >> 26); h2 = (uint32_t) d2 & 0x3ffffff;
            d3 += c; c = (uint32_t) (d3 >> 26); h3 = (uint32_t) d3 & 0x3ffffff;
            d4 += c; c = (uint32_t) (d4 >> 26); h4 = (uint32_t) d4 & 0x3ffffff;
            h0 += c * 5; c = h0 >> 26; h0 &= 0x3ffffff;
            h1 += c;
        }
        h[0] = h0; h[1] = h1; h[2] = h2; h[3] = h3; h[4] = h4;
    }
    uint32_t r[5] = {}, h[5] = {}, pad[4] = {};
    uint8_t  buf[16] = {};
    size_t   lleno = 0;
};

namespace detalle {

inline void etiqueta(const uint8_t key[32], const uint8_t nonce[12], const uint8_t * aad, size_t n_aad,
                     const uint8_t * c, size_t n_c, uint8_t tag[16]) {
    uint8_t otk[64];
    bloque(key, 0, nonce, otk);   // clave de Poly1305 = primeros 32 bytes del bloque 0
    poly1305 p(otk);
    static const uint8_t ceros[16] = {};
    p.actualizar(aad, n_aad);
    if (n_aad % 16) p.actualizar(ceros, 16 - n_aad % 16);
    p.actualizar(c, n_c);
    if (n_c % 16) p.actualizar(ceros, 16 - n_c % 16);
    uint8_t largos[16];
    for (int i = 0; i < 8; ++i) { largos[i] = (uint8_t) ((uint64_t) n_aad >> (8 * i)); largos[8 + i] = (uint8_t) ((uint64_t) n_c >> (8 * i)); }
    p.actualizar(largos, 16);
    p.terminar(tag);
}

} // namespace detalle

// AEAD (RFC 8439 §2.8): cifra `datos` en su lugar y devuelve la etiqueta.
inline void sellar(const uint8_t key[32], const uint8_t nonce[12], const uint8_t * aad, size_t n_aad,
                   uint8_t * datos, size_t n, uint8_t tag[16]) {
    chacha20(key, 1, nonce, datos, n);
    detalle::etiqueta(key, nonce, aad, n_aad, datos, n, tag);
}

// Verifica la etiqueta (tiempo constante) y descifra en su lugar. false = alterado o clave incorrecta.
inline bool abrir(const uint8_t key[32], const uint8_t nonce[12], const uint8_t * aad, size_t n_aad,
                  uint8_t * datos, size_t n, const uint8_t tag[16]) {
    uint8_t esperado[16];
    detalle::etiqueta(key, nonce, aad, n_aad, datos, n, esperado);
    uint8_t dif = 0;
    for (int i = 0; i < 16; ++i) dif |= (uint8_t) (esperado[i] ^ tag[i]);
    if (dif != 0) return false;
    chacha20(key, 1, nonce, datos, n);
    return true;
}

inline bool es_cifrado(const std::string & b) { return b.size() >= 33 && b.compare(0, 4, "NAIC") == 0; }

// Cifra un archivo completo. `contexto` (p. ej. el nombre del archivo) va en los datos asociados:
// un archivo cifrado copiado con otro nombre no se descifra.
inline std::string cifrar(const clave & c, const std::string & plano, const std::string & contexto) {
    std::string out = "NAIC";
    out += (char) 1;
    uint8_t nonce[12];
    std::random_device rd;
    for (int i = 0; i < 12; i += 4) detalle::pon32(nonce + i, rd());
    out.append((const char *) nonce, 12);
    std::string datos = plano;
    const std::string aad = out.substr(0, 5) + contexto;
    uint8_t tag[16];
    sellar(c.k, nonce, (const uint8_t *) aad.data(), aad.size(), (uint8_t *) datos.data(), datos.size(), tag);
    out += datos;
    out.append((const char *) tag, 16);
    return out;
}

inline bool descifrar(const clave & c, const std::string & archivo, const std::string & contexto, std::string & plano) {
    if (!es_cifrado(archivo) || archivo[4] != 1) return false;
    const uint8_t * nonce = (const uint8_t *) archivo.data() + 5;
    plano.assign(archivo, 17, archivo.size() - 33);
    const uint8_t * tag = (const uint8_t *) archivo.data() + archivo.size() - 16;
    const std::string aad = archivo.substr(0, 5) + contexto;
    if (!abrir(c.k, nonce, (const uint8_t *) aad.data(), aad.size(), (uint8_t *) plano.data(), plano.size(), tag)) {
        plano.clear();
        return false;
    }
    return true;
}

// Lee la clave (64 caracteres hex) o la crea con 256 bits aleatorios y permisos 0600.
inline bool cargar_o_crear(const std::string & ruta, clave & c, std::string & aviso) {
    auto hex = [](char ch) -> int {
        if (ch >= '0' && ch <= '9') return ch - '0';
        if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
        if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
        return -1;
    };
    std::ifstream f(ruta);
    if (f) {
        std::string s, l;
        while (std::getline(f, l)) for (char ch : l) if (hex(ch) >= 0) s += ch;
        if (s.size() != 64) { aviso = "la clave de cifrado debe tener 64 caracteres hexadecimales (32 bytes)"; return false; }
        for (int i = 0; i < 32; ++i) c.k[i] = (uint8_t) (hex(s[2 * i]) * 16 + hex(s[2 * i + 1]));
        c.activa = true;
        return true;
    }
    std::random_device rd;
    for (int i = 0; i < 32; i += 4) detalle::pon32(c.k + i, rd());
    static const char * H = "0123456789abcdef";
    std::string s;
    for (int i = 0; i < 32; ++i) { s += H[c.k[i] >> 4]; s += H[c.k[i] & 15]; }
    {
        std::ofstream o(ruta, std::ios::trunc);
        if (!o || !(o << s << "\n")) { aviso = "no se pudo crear la clave de cifrado en " + ruta; return false; }
    }
    std::error_code ec;
    std::filesystem::permissions(ruta, std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
                                 std::filesystem::perm_options::replace, ec);
    c.activa = true;
    aviso = "se creó una clave de cifrado nueva en " + ruta + ": respáldala en un lugar seguro; sin ella las sesiones cifradas no se recuperan";
    return true;
}

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4), HMAC (RFC 2104) y PBKDF2 (RFC 8018) para el login: contraseñas guardadas
// como PBKDF2-HMAC-SHA256 con sal, y tokens guardados solo como su SHA-256. Verificado con los
// vectores de FIPS, RFC 4231 y RFC 7914 en tests/test-nai.cpp.
// ---------------------------------------------------------------------------
class sha256 {
public:
    sha256() { reiniciar(); }
    void reiniciar() {
        static const uint32_t H0[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
        memcpy(h, H0, sizeof(h));
        n = 0; total = 0;
    }
    void actualizar(const void * datos, size_t len) {
        const uint8_t * p = (const uint8_t *) datos;
        total += len;
        while (len > 0) {
            const size_t t = 64 - n < len ? 64 - n : len;
            memcpy(buf + n, p, t);
            n += t; p += t; len -= t;
            if (n == 64) { bloque(buf); n = 0; }
        }
    }
    void terminar(uint8_t out[32]) {
        const uint64_t bits = total * 8;
        const uint8_t uno = 0x80, cero = 0;
        actualizar(&uno, 1);
        while (n != 56) actualizar(&cero, 1);
        uint8_t l[8];
        for (int i = 0; i < 8; ++i) l[i] = (uint8_t) (bits >> (56 - 8 * i));
        actualizar(l, 8);
        for (int i = 0; i < 8; ++i) {
            out[4 * i] = (uint8_t) (h[i] >> 24); out[4 * i + 1] = (uint8_t) (h[i] >> 16);
            out[4 * i + 2] = (uint8_t) (h[i] >> 8); out[4 * i + 3] = (uint8_t) h[i];
        }
    }
private:
    static uint32_t ror(uint32_t x, int c) { return (x >> c) | (x << (32 - c)); }
    void bloque(const uint8_t * b) {
        static const uint32_t K[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2 };
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) w[i] = (uint32_t) b[4 * i] << 24 | (uint32_t) b[4 * i + 1] << 16 | (uint32_t) b[4 * i + 2] << 8 | b[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], bb = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = hh + (ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
            const uint32_t t2 = (ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)) + ((a & bb) ^ (a & c) ^ (bb & c));
            hh = g; g = f; f = e; e = d + t1; d = c; c = bb; bb = a; a = t1 + t2;
        }
        h[0] += a; h[1] += bb; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    uint32_t h[8];
    uint8_t  buf[64];
    size_t   n = 0;
    uint64_t total = 0;
};

inline std::string sha256_de(const std::string & s) {
    sha256 x;
    x.actualizar(s.data(), s.size());
    uint8_t o[32];
    x.terminar(o);
    return std::string((const char *) o, 32);
}

inline void hmac_sha256(const uint8_t * clave_, size_t n_clave, const uint8_t * msg, size_t n_msg, uint8_t out[32]) {
    uint8_t k[64] = {};
    if (n_clave > 64) { sha256 x; x.actualizar(clave_, n_clave); x.terminar(k); }
    else memcpy(k, clave_, n_clave);
    uint8_t ip[64], op[64];
    for (int i = 0; i < 64; ++i) { ip[i] = k[i] ^ 0x36; op[i] = k[i] ^ 0x5c; }
    uint8_t in[32];
    sha256 a; a.actualizar(ip, 64); a.actualizar(msg, n_msg); a.terminar(in);
    sha256 b; b.actualizar(op, 64); b.actualizar(in, 32); b.terminar(out);
}

// PBKDF2-HMAC-SHA256. Las claves internas (ipad/opad) se precalculan una vez: cada iteración
// cuesta dos compresiones de SHA-256, no cuatro.
inline void pbkdf2_sha256(const std::string & pass, const uint8_t * sal, size_t n_sal, uint32_t iter, uint8_t * out, size_t n_out) {
    uint8_t k[64] = {};
    if (pass.size() > 64) { sha256 x; x.actualizar(pass.data(), pass.size()); x.terminar(k); }
    else memcpy(k, pass.data(), pass.size());
    uint8_t ip[64], op[64];
    for (int i = 0; i < 64; ++i) { ip[i] = k[i] ^ 0x36; op[i] = k[i] ^ 0x5c; }
    sha256 base_i, base_o;
    base_i.actualizar(ip, 64);
    base_o.actualizar(op, 64);
    auto prf = [&](const uint8_t * m, size_t nm, uint8_t r[32]) {
        uint8_t in[32];
        sha256 a = base_i; a.actualizar(m, nm); a.terminar(in);
        sha256 b = base_o; b.actualizar(in, 32); b.terminar(r);
    };
    for (uint32_t bloque_ = 1; n_out > 0; ++bloque_) {
        std::string m((const char *) sal, n_sal);
        m += (char) (bloque_ >> 24); m += (char) (bloque_ >> 16); m += (char) (bloque_ >> 8); m += (char) bloque_;
        uint8_t u[32], t[32];
        prf((const uint8_t *) m.data(), m.size(), u);
        memcpy(t, u, 32);
        for (uint32_t i = 1; i < iter; ++i) {
            prf(u, 32, u);
            for (int j = 0; j < 32; ++j) t[j] ^= u[j];
        }
        const size_t c = n_out < 32 ? n_out : 32;
        memcpy(out, t, c);
        out += c; n_out -= c;
    }
}

// Bytes aleatorios del sistema (random_device usa /dev/urandom o el generador del sistema operativo).
inline void aleatorio(uint8_t * out, size_t n) {
    std::random_device rd;
    for (size_t i = 0; i < n; i += 4) {
        const uint32_t v = rd();
        for (size_t j = 0; j < 4 && i + j < n; ++j) out[i + j] = (uint8_t) (v >> (8 * j));
    }
}

inline bool iguales_tiempo_constante(const std::string & a, const std::string & b) {
    if (a.size() != b.size()) return false;
    uint8_t d = 0;
    for (size_t i = 0; i < a.size(); ++i) d |= (uint8_t) (a[i] ^ b[i]);
    return d == 0;
}

inline std::string a_hex(const std::string & b) {
    static const char * H = "0123456789abcdef";
    std::string s;
    for (unsigned char c : b) { s += H[c >> 4]; s += H[c & 15]; }
    return s;
}

inline std::string de_hex(const std::string & s) {
    auto v = [](char ch) -> int {
        if (ch >= '0' && ch <= '9') return ch - '0';
        if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
        if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
        return -1;
    };
    std::string r;
    for (size_t i = 0; i + 1 < s.size(); i += 2) {
        const int a = v(s[i]), b = v(s[i + 1]);
        if (a < 0 || b < 0) return "";
        r += (char) (a * 16 + b);
    }
    return r;
}

// base64url sin relleno (tokens que viajan en cabeceras HTTP)
inline std::string base64url(const uint8_t * d, size_t n) {
    static const char * A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    std::string r;
    for (size_t i = 0; i < n; i += 3) {
        const uint32_t v = (uint32_t) d[i] << 16 | (i + 1 < n ? (uint32_t) d[i + 1] << 8 : 0) | (i + 2 < n ? d[i + 2] : 0);
        r += A[(v >> 18) & 63]; r += A[(v >> 12) & 63];
        if (i + 1 < n) r += A[(v >> 6) & 63];
        if (i + 2 < n) r += A[v & 63];
    }
    return r;
}

} // namespace nai_cifrado
