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

} // namespace nai_cifrado
