// [NAI] Lector de documentos. Ver nai-lector.h.
#include "nai-lector.h"

#include "http.h"
#include "subproc.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <vector>

#ifndef _WIN32
#include <netdb.h>
#endif

namespace nai_lector {

static std::string minusculas(std::string s) {
    for (auto & c : s) c = (char) std::tolower((unsigned char) c);
    return s;
}

static std::string extension(const std::string & nombre) {
    const auto p = nombre.rfind('.');
    if (p == std::string::npos) return "";
    return minusculas(nombre.substr(p + 1));
}

static bool utf8_valido(const std::string & s) {
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = (unsigned char) s[i];
        if (c < 0x80) { ++i; continue; }
        const int k = (c >> 5) == 6 ? 1 : (c >> 4) == 14 ? 2 : (c >> 3) == 30 ? 3 : -1;
        if (k < 0 || i + (size_t) k >= s.size()) return false;
        for (int j = 1; j <= k; ++j) {
            if (((unsigned char) s[i + j] >> 6) != 2) return false;
        }
        i += k + 1;
    }
    return true;
}

static std::string latin1_a_utf8(const std::string & s) {
    std::string r;
    for (unsigned char c : s) {
        if (c < 0x80) r += (char) c;
        else { r += (char) (0xC0 | (c >> 6)); r += (char) (0x80 | (c & 0x3F)); }
    }
    return r;
}

static std::string a_utf8(const std::string & s) {
    std::string t = s;
    if (t.size() >= 3 && (unsigned char) t[0] == 0xEF && (unsigned char) t[1] == 0xBB && (unsigned char) t[2] == 0xBF) t.erase(0, 3);
    return utf8_valido(t) ? t : latin1_a_utf8(t);
}

static void agregar_utf8(std::string & out, uint32_t cp) {
    if (cp < 0x80) out += (char) cp;
    else if (cp < 0x800) { out += (char) (0xC0 | (cp >> 6)); out += (char) (0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { out += (char) (0xE0 | (cp >> 12)); out += (char) (0x80 | ((cp >> 6) & 0x3F)); out += (char) (0x80 | (cp & 0x3F)); }
    else if (cp < 0x110000) {
        out += (char) (0xF0 | (cp >> 18)); out += (char) (0x80 | ((cp >> 12) & 0x3F));
        out += (char) (0x80 | ((cp >> 6) & 0x3F)); out += (char) (0x80 | (cp & 0x3F));
    }
}

// Bloque <etiqueta ...>...</etiqueta> más grande (para quedarse con el contenido principal).
static std::string bloque_mayor(const std::string & html, const std::string & low, const std::string & etiqueta) {
    std::string mejor;
    size_t i = 0;
    const std::string ab = "<" + etiqueta, ci = "</" + etiqueta + ">";
    while ((i = low.find(ab, i)) != std::string::npos) {
        const char sig = i + ab.size() < low.size() ? low[i + ab.size()] : '>';
        if (sig != '>' && sig != ' ' && sig != '\t' && sig != '\n' && sig != '\r') { i += ab.size(); continue; }
        const size_t f = low.find(ci, i);
        if (f == std::string::npos) break;
        if (f + ci.size() - i > mejor.size()) mejor = html.substr(i, f + ci.size() - i);
        i = f + ci.size();
    }
    return mejor;
}

static std::string html_a_texto_bruto(const std::string & html0);

std::string html_a_texto(const std::string & html0) {
    // Páginas con <article> o <main>: usar solo el contenido principal si tiene texto suficiente.
    const std::string html = a_utf8(html0);
    const std::string low  = minusculas(html);
    for (const char * et : { "article", "main" }) {
        const std::string b = bloque_mayor(html, low, et);
        if (b.size() > 0) {
            const std::string t = html_a_texto_bruto(b);
            if (t.size() >= 500) return t;
        }
    }
    return html_a_texto_bruto(html);
}

// Quita restos de navegación: líneas de menos de 6 palabras (botones, enlaces, migas de pan).
std::string limpiar_web(const std::string & texto) {
    std::istringstream in(texto);
    std::string l, r;
    bool vacia = false;
    while (std::getline(in, l)) {
        int palabras = 0;
        bool en = false;
        for (unsigned char c : l) {
            const bool letra = std::isalnum(c) || c >= 0x80;
            if (letra && !en) palabras++;
            en = letra;
        }
        if (palabras >= 6) {
            if (vacia && !r.empty()) r += "\n"; // separar párrafos solo entre líneas conservadas
            r += l + "\n";
            vacia = false;
        } else {
            vacia = true;
        }
    }
    return r;
}

static std::string html_a_texto_bruto(const std::string & html0) {
    const std::string html = a_utf8(html0);
    const std::string low  = minusculas(html);
    std::string out;
    size_t i = 0;
    static const char * bloques_fuera[] = { "script", "style", "noscript", "svg", "head", "template", "iframe",
                                            "nav", "footer", "aside", "form", "button", "select", "dialog" };
    static const char * bloques_linea[] = { "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
                                            "section", "article", "blockquote", "pre", "table", "ul", "ol", "header", "footer" };
    while (i < html.size()) {
        if (html[i] == '<') {
            // comentario
            if (low.compare(i, 4, "<!--") == 0) {
                const size_t f = low.find("-->", i + 4);
                i = f == std::string::npos ? html.size() : f + 3;
                continue;
            }
            size_t j = i + 1;
            const bool cierre = j < html.size() && html[j] == '/';
            if (cierre) ++j;
            size_t k = j;
            while (k < html.size() && std::isalnum((unsigned char) html[k])) ++k;
            const std::string nombre = low.substr(j, k - j);
            const size_t fin = html.find('>', i);
            if (fin == std::string::npos) break;
            bool saltado = false;
            if (!cierre) {
                for (const char * b : bloques_fuera) {
                    if (nombre == b) {
                        const size_t f = low.find(std::string("</") + b, fin);
                        const size_t g = f == std::string::npos ? std::string::npos : low.find('>', f);
                        i = g == std::string::npos ? html.size() : g + 1;
                        saltado = true;
                        break;
                    }
                }
            }
            if (saltado) continue;
            bool linea = false;
            for (const char * b : bloques_linea) if (nombre == b) { linea = true; break; }
            out += linea ? "\n\n" : " ";
            i = fin + 1;
            continue;
        }
        if (html[i] == '&') {
            const size_t f = html.find(';', i);
            if (f != std::string::npos && f - i <= 10) {
                const std::string ent = html.substr(i + 1, f - i - 1);
                static const std::pair<const char *, uint32_t> tabla[] = {
                    { "amp", '&' }, { "lt", '<' }, { "gt", '>' }, { "quot", '"' }, { "apos", '\'' }, { "nbsp", ' ' },
                    { "aacute", 0xE1 }, { "eacute", 0xE9 }, { "iacute", 0xED }, { "oacute", 0xF3 }, { "uacute", 0xFA },
                    { "ntilde", 0xF1 }, { "Ntilde", 0xD1 }, { "uuml", 0xFC }, { "iexcl", 0xA1 }, { "iquest", 0xBF },
                    { "laquo", 0xAB }, { "raquo", 0xBB }, { "mdash", 0x2014 }, { "ndash", 0x2013 }, { "hellip", 0x2026 },
                    { "ldquo", 0x201C }, { "rdquo", 0x201D }, { "lsquo", 0x2018 }, { "rsquo", 0x2019 }, { "copy", 0xA9 },
                };
                uint32_t cp = 0;
                if (!ent.empty() && ent[0] == '#') {
                    try { cp = (ent.size() > 1 && (ent[1] == 'x' || ent[1] == 'X')) ? std::stoul(ent.substr(2), nullptr, 16) : std::stoul(ent.substr(1)); }
                    catch (...) { cp = 0; }
                } else {
                    for (auto & t : tabla) if (ent == t.first) { cp = t.second; break; }
                }
                if (cp) { agregar_utf8(out, cp); i = f + 1; continue; }
            }
        }
        out += html[i++];
    }
    // espacios: una línea vacía entre párrafos
    std::string r, linea;
    std::istringstream in(out);
    int vacias = 0;
    while (std::getline(in, linea)) {
        std::string l;
        bool esp = false;
        for (unsigned char c : linea) {
            if (c == ' ' || c == '\t' || c == '\r') { esp = true; continue; }
            if (esp && !l.empty()) l += ' ';
            esp = false;
            l += (char) c;
        }
        if (l.empty()) { if (++vacias == 1 && !r.empty()) r += "\n"; continue; }
        vacias = 0;
        r += l + "\n";
    }
    return r;
}

// Ejecuta un convertidor externo sin shell (argumentos separados) y devuelve su salida.
static bool ejecutar(const std::vector<std::string> & args, std::string & salida, std::string & error) {
    if (!common_subproc::is_supported()) { error = "este binario se compiló sin soporte de subprocesos (LLAMA_SUBPROCESS)"; return false; }
    common_subproc p;
    if (!p.create(args, subprocess_option_search_user_path | subprocess_option_inherit_environment | subprocess_option_no_window)) {
        error = "no se encontró '" + args[0] + "'";
        return false;
    }
    p.close_stdin();
    salida.clear();
    char buf[65536];
    FILE * f = p.stdout_file();
    while (f) {
        const size_t n = fread(buf, 1, sizeof(buf), f);
        if (n == 0) break;
        salida.append(buf, n);
        if (salida.size() > (64u << 20)) break;
    }
    const int rc = p.join();
    if (rc != 0) { error = "'" + args[0] + "' terminó con código " + std::to_string(rc); return false; }
    return true;
}

static std::string ruta_temporal(const std::string & ext) {
    std::random_device rd;
    char nombre[64];
    snprintf(nombre, sizeof(nombre), "nai-lector-%08x%08x.%s", rd(), rd(), ext.c_str());
    return (std::filesystem::temp_directory_path() / nombre).string();
}

static resultado convertir_externo(const std::string & contenido, const std::string & ext) {
    resultado r;
    r.tipo = ext;
    const std::string tmp = ruta_temporal(ext);
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f.write(contenido.data(), (std::streamsize) contenido.size());
    }
    std::error_code ec;
    std::filesystem::permissions(tmp, std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
                                 std::filesystem::perm_options::replace, ec);
    std::string salida, error;
    bool ok = false;
    if (ext == "pdf") {
        ok = ejecutar({ "pdftotext", "-enc", "UTF-8", "-layout", tmp, "-" }, salida, error);
        if (!ok) error += " (instala poppler: brew install poppler / apt install poppler-utils)";
    } else {
        ok = ejecutar({ "textutil", "-convert", "txt", "-encoding", "UTF-8", "-stdout", tmp }, salida, error);
        if (!ok) {
            std::string e2;
            ok = ejecutar({ "pandoc", "-t", "plain", "--wrap=none", tmp }, salida, e2);
            if (!ok) error = "se necesita textutil (macOS) o pandoc para leer ." + ext;
        }
    }
    std::filesystem::remove(tmp, ec);
    if (!ok) { r.error = error; return r; }
    r.texto = a_utf8(salida);
    r.ok = !r.texto.empty();
    if (!r.ok) r.error = "el documento no tiene texto extraíble (¿es una imagen escaneada?)";
    return r;
}

resultado leer_bytes(const std::string & nombre, const std::string & contenido, const std::string & content_type) {
    resultado r;
    std::string ext = extension(nombre);
    const std::string ct = minusculas(content_type);
    if (ext.empty() || ext.size() > 5) {
        if (ct.find("html") != std::string::npos)      ext = "html";
        else if (ct.find("pdf") != std::string::npos)  ext = "pdf";
        else if (ct.find("wordprocessingml") != std::string::npos) ext = "docx";
        else if (ct.find("rtf") != std::string::npos)  ext = "rtf";
        else                                           ext = "txt";
    }
    if (contenido.size() >= 5 && contenido.compare(0, 5, "%PDF-") == 0) ext = "pdf";
    if (ext == "pdf" || ext == "docx" || ext == "doc" || ext == "rtf" || ext == "odt" || ext == "pages") {
        return convertir_externo(contenido, ext);
    }
    if (ext == "htm" || ext == "html" || ext == "xhtml" || ct.find("html") != std::string::npos) {
        r.tipo = "html";
        r.texto = limpiar_web(html_a_texto(contenido));
    } else {
        // texto, md, csv, json, código...: si hay bytes nulos es binario
        if (std::find(contenido.begin(), contenido.begin() + std::min<size_t>(contenido.size(), 8192), '\0') !=
            contenido.begin() + std::min<size_t>(contenido.size(), 8192)) {
            r.error = "formato binario no soportado: ." + ext;
            return r;
        }
        r.tipo = ext;
        r.texto = a_utf8(contenido);
    }
    r.ok = !r.texto.empty();
    if (!r.ok) r.error = "el documento está vacío";
    return r;
}

resultado leer_archivo(const std::string & ruta) {
    std::ifstream f(ruta, std::ios::binary);
    if (!f) { resultado r; r.error = "no se pudo abrir " + ruta; return r; }
    std::string b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return leer_bytes(ruta, b);
}

// ---------------------------------------------------------------------------
// URLs
// ---------------------------------------------------------------------------
static bool ip_publica(const sockaddr * sa) {
    if (sa->sa_family == AF_INET) {
        const uint8_t * b = (const uint8_t *) &((const sockaddr_in *) sa)->sin_addr;
        if (b[0] == 0 || b[0] == 10 || b[0] == 127) return false;
        if (b[0] == 100 && (b[1] & 0xC0) == 64) return false;   // 100.64/10 (CGNAT)
        if (b[0] == 169 && b[1] == 254) return false;            // enlace local y metadatos de nube
        if (b[0] == 172 && (b[1] & 0xF0) == 16) return false;
        if (b[0] == 192 && b[1] == 168) return false;
        if (b[0] >= 224) return false;                           // multicast y reservadas
        return true;
    }
    if (sa->sa_family == AF_INET6) {
        const uint8_t * b = (const uint8_t *) &((const sockaddr_in6 *) sa)->sin6_addr;
        static const uint8_t cero[16] = {};
        if (memcmp(b, cero, 15) == 0 && (b[15] == 0 || b[15] == 1)) return false; // :: y ::1
        if ((b[0] & 0xFE) == 0xFC) return false;                                   // fc00::/7
        if (b[0] == 0xFE && (b[1] & 0xC0) == 0x80) return false;                   // fe80::/10
        if (b[0] == 0xFF) return false;                                            // multicast
        static const uint8_t v4map[12] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF };
        if (memcmp(b, v4map, 12) == 0) {
            sockaddr_in s4 = {};
            s4.sin_family = AF_INET;
            memcpy(&s4.sin_addr, b + 12, 4);
            return ip_publica((const sockaddr *) &s4);
        }
        return true;
    }
    return false;
}

// Resuelve el host y devuelve una IP permitida (texto) o vacío con el error.
static std::string resolver(const std::string & host, bool permitir_local, std::string & error) {
#ifdef _WIN32
    (void) permitir_local;
    return host; // en Windows no se valida (sin getaddrinfo aquí)
#else
    addrinfo hints = {};
    hints.ai_socktype = SOCK_STREAM;
    addrinfo * res = nullptr;
    if (getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || !res) { error = "no se pudo resolver " + host; return ""; }
    std::string ip;
    for (addrinfo * a = res; a; a = a->ai_next) {
        if (!permitir_local && !ip_publica(a->ai_addr)) {
            error = "la URL apunta a una dirección interna; se rechaza por seguridad (usa --nai-lector-red-local si es tu red)";
            ip.clear();
            break;
        }
        if (ip.empty()) {
            char buf[INET6_ADDRSTRLEN] = {};
            const void * src = a->ai_family == AF_INET ? (const void *) &((sockaddr_in *) a->ai_addr)->sin_addr
                                                       : (const void *) &((sockaddr_in6 *) a->ai_addr)->sin6_addr;
            if (inet_ntop(a->ai_family, src, buf, sizeof(buf))) ip = buf;
        }
    }
    freeaddrinfo(res);
    if (ip.empty() && error.empty()) error = "sin direcciones para " + host;
    return ip;
#endif
}

resultado leer_url(const std::string & url0, bool permitir_local, size_t max_bytes) {
    resultado r;
    std::string url = url0;
    for (int salto = 0; salto < 5; ++salto) {
        try {
            auto [cli, partes] = common_http_client(url);
            std::string error;
            const std::string ip = resolver(partes.host, permitir_local, error);
            if (ip.empty()) { r.error = error; return r; }
            cli.set_hostname_addr_map({ { partes.host, ip } }); // conectar a la IP validada
            cli.set_follow_location(false);
            cli.set_connection_timeout(10, 0);
            cli.set_read_timeout(30, 0);
            std::string cuerpo;
            bool excedido = false;
            httplib::Headers hdr = { { "User-Agent", "NAI-lector/1.0" }, { "Accept", "text/html,text/plain,application/pdf,*/*;q=0.5" } };
            auto res = cli.Get(partes.path, hdr, [&](const char * d, size_t n) {
                if (cuerpo.size() + n > max_bytes) { excedido = true; return false; }
                cuerpo.append(d, n);
                return true;
            });
            if (excedido) { r.error = "el documento supera " + std::to_string(max_bytes >> 20) + " MiB"; return r; }
            if (!res) { r.error = "no se pudo descargar: " + httplib::to_string(res.error()); return r; }
            if (res->status >= 300 && res->status < 400 && res->has_header("Location")) {
                std::string loc = res->get_header_value("Location");
                if (loc.rfind("http://", 0) != 0 && loc.rfind("https://", 0) != 0) {
                    const std::string base = partes.scheme + "://" + common_http_format_host(partes.host) + ":" + std::to_string(partes.port);
                    if (!loc.empty() && loc[0] == '/') loc = base + loc;
                    else loc = base + partes.path.substr(0, partes.path.rfind('/') + 1) + loc;
                }
                url = loc;
                continue;
            }
            if (res->status != 200) { r.error = "el servidor respondió " + std::to_string(res->status); return r; }
            std::string nombre = partes.path;
            const auto q = nombre.find_first_of("?#");
            if (q != std::string::npos) nombre.resize(q);
            return leer_bytes(nombre, cuerpo, res->get_header_value("Content-Type"));
        } catch (const std::exception & e) {
            r.error = e.what();
            return r;
        }
    }
    r.error = "demasiadas redirecciones";
    return r;
}

} // namespace nai_lector
