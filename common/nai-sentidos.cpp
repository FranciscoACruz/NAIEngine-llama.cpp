// [NAI] Sentidos: ver imágenes y videos, oír audio. Ver nai-sentidos.h.
#include "nai-sentidos.h"

#include "base64.hpp"
#include "http.h"
#include "nai-cifrado.h"
#include "nai-lector.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <vector>

using json = nlohmann::ordered_json;

namespace nai_sentidos {

static std::mutex mtx_cfg;
static config     cfg_actual;

void configurar(const config & c) {
    std::lock_guard<std::mutex> lk(mtx_cfg);
    cfg_actual = c;
    if (cfg_actual.cuadros < 1)  cfg_actual.cuadros = 1;
    if (cfg_actual.cuadros > 16) cfg_actual.cuadros = 16;
}

const config & actual() { return cfg_actual; }   // se configura una vez al arrancar

// ---------------------------------------------------------------------------
// Tipo de medio
// ---------------------------------------------------------------------------

static std::string minus(std::string s) {
    for (auto & c : s) c = (char) std::tolower((unsigned char) c);
    return s;
}

static std::string ext_de(const std::string & nombre) {
    const auto barra = nombre.find_last_of("/\\");
    const auto punto = nombre.rfind('.');
    if (punto == std::string::npos || (barra != std::string::npos && punto < barra)) return "";
    return minus(nombre.substr(punto + 1));
}

static bool empieza(const std::string & b, size_t pos, const char * s) {
    const size_t n = std::char_traits<char>::length(s);
    return b.size() >= pos + n && b.compare(pos, n, s) == 0;
}

// por los primeros bytes; "" si no se reconoce
static std::string mime_magico(const std::string & b) {
    if (empieza(b, 0, "\x89PNG"))                           return "image/png";
    if (b.size() > 3 && (unsigned char) b[0] == 0xFF && (unsigned char) b[1] == 0xD8 && (unsigned char) b[2] == 0xFF) return "image/jpeg";
    if (empieza(b, 0, "GIF8"))                              return "image/gif";
    if (empieza(b, 0, "RIFF") && empieza(b, 8, "WEBP"))     return "image/webp";
    if (empieza(b, 0, "RIFF") && empieza(b, 8, "WAVE"))     return "audio/wav";
    if (empieza(b, 0, "RIFF") && empieza(b, 8, "AVI "))     return "video/x-msvideo";
    if (empieza(b, 0, "fLaC"))                              return "audio/flac";
    if (empieza(b, 0, "OggS"))                              return "audio/ogg";
    if (empieza(b, 0, "ID3"))                               return "audio/mpeg";
    if (b.size() > 2 && (unsigned char) b[0] == 0xFF && ((unsigned char) b[1] & 0xE0) == 0xE0) return "audio/mpeg";
    if (empieza(b, 0, "\x1A\x45\xDF\xA3"))                  return "video/webm";
    if (empieza(b, 4, "ftyp")) {
        const std::string marca = b.size() >= 12 ? b.substr(8, 4) : "";
        if (marca == "M4A " || marca == "M4B ")             return "audio/mp4";
        if (marca == "qt  ")                                return "video/quicktime";
        return "video/mp4";
    }
    if (empieza(b, 0, "BM") && b.size() > 26)               return "image/bmp";
    return "";
}

static std::string tipo_de_mime(const std::string & m) {
    if (m.rfind("image/", 0) == 0) return "imagen";
    if (m.rfind("audio/", 0) == 0) return "audio";
    if (m.rfind("video/", 0) == 0) return "video";
    return "";
}

std::string tipo_medio(const std::string & nombre, const std::string & content_type, const std::string & bytes) {
    const std::string ct  = minus(content_type);
    const std::string mag = mime_magico(bytes);
    // un .webm o .ogg puede ser solo audio: manda el Content-Type o la extensión
    if (!mag.empty() && (mag == "video/webm" || mag == "audio/ogg")) {
        const std::string e = ext_de(nombre);
        if (ct.rfind("audio/", 0) == 0 || e == "weba" || e == "oga" || e == "opus" || e == "ogg") return "audio";
        if (ct.rfind("video/", 0) == 0 || e == "webm" || e == "ogv" || e == "mkv") return "video";
        return tipo_de_mime(mag);
    }
    if (!mag.empty()) return tipo_de_mime(mag);
    const std::string t = tipo_de_mime(ct);
    if (!t.empty() && ct.find("svg") == std::string::npos) return t;
    static const char * img[] = { "png", "jpg", "jpeg", "gif", "webp", "bmp" };
    static const char * aud[] = { "wav", "mp3", "flac", "ogg", "oga", "opus", "m4a", "aac", "weba" };
    static const char * vid[] = { "mp4", "mov", "m4v", "webm", "mkv", "avi", "ogv", "3gp" };
    const std::string e = ext_de(nombre);
    for (auto x : img) if (e == x) return "imagen";
    for (auto x : aud) if (e == x) return "audio";
    for (auto x : vid) if (e == x) return "video";
    return "";
}

std::string mime_de(const std::string & tipo, const std::string & bytes) {
    const std::string m = mime_magico(bytes);
    if (!m.empty() && tipo_de_mime(m) == tipo) return m;
    if (tipo == "imagen") return "image/png";
    if (tipo == "audio")  return m.empty() ? "audio/wav" : m;
    return "video/mp4";
}

// ---------------------------------------------------------------------------
// Llamadas a los sentidos
// ---------------------------------------------------------------------------

static std::string ruta_chat(std::string path) {
    while (!path.empty() && path.back() == '/') path.pop_back();
    if (path.size() >= 17 && path.compare(path.size() - 17, 17, "/chat/completions") == 0) return path;
    if (path.size() >= 3 && path.compare(path.size() - 3, 3, "/v1") == 0) return path + "/chat/completions";
    return path + "/v1/chat/completions";
}

static std::string sin_pensamiento(std::string t) {
    for (size_t a; (a = t.find("<think>")) != std::string::npos;) {
        const size_t b = t.find("</think>", a);
        t.erase(a, b == std::string::npos ? std::string::npos : b + 8 - a);
    }
    const size_t i = t.find_first_not_of(" \t\r\n");
    if (i == std::string::npos) return "";
    return t.substr(i, t.find_last_not_of(" \t\r\n") - i + 1);
}

static httplib::Headers cabeceras() {
    httplib::Headers h = { { "X-NAI-Sentido", "1" } };   // el servidor NAI no le aplica memoria ni sentidos a esta petición
    if (!actual().clave.empty()) h.emplace("Authorization", "Bearer " + actual().clave);
    return h;
}

static bool pedir_chat(const std::string & base, const json & contenido, const std::string & sistema, int max_tokens,
                       std::string & texto, std::string & error) {
    try {
        auto [cli, partes] = common_http_client(base);
        cli.set_connection_timeout(5, 0);
        cli.set_read_timeout(300, 0);
        cli.set_write_timeout(120, 0);
        json msgs = json::array();
        if (!sistema.empty()) msgs.push_back({ { "role", "system" }, { "content", sistema } });
        msgs.push_back({ { "role", "user" }, { "content", contenido } });
        const json cuerpo = {
            { "messages", msgs }, { "max_tokens", max_tokens }, { "temperature", 0.2 }, { "stream", false },
            { "chat_template_kwargs", { { "enable_thinking", false } } },
        };
        auto res = cli.Post(ruta_chat(partes.path), cabeceras(), cuerpo.dump(), "application/json");
        if (!res) { error = "no responde " + base + " (" + httplib::to_string(res.error()) + ")"; return false; }
        if (res->status != 200) {
            std::string msg = res->body.substr(0, 300);
            try { msg = json::parse(res->body).at("error").value("message", msg); } catch (...) {}
            error = base + " respondió " + std::to_string(res->status) + ": " + msg;
            return false;
        }
        const json r = json::parse(res->body);
        texto = sin_pensamiento(r.at("choices").at(0).at("message").value("content", std::string()));
        if (texto.empty()) { error = base + " no devolvió texto"; return false; }
        return true;
    } catch (const std::exception & e) {
        error = std::string("sentido: ") + e.what();
        return false;
    }
}

// whisper.cpp server: POST multipart a .../inference -> {"text": "..."}
static bool pedir_whisper(const std::string & url, const std::string & bytes, const std::string & nombre, std::string & texto,
                          std::string & error) {
    try {
        auto [cli, partes] = common_http_client(url);
        cli.set_connection_timeout(5, 0);
        cli.set_read_timeout(600, 0);
        httplib::UploadFormDataItems items = {
            { "file", bytes, nombre, "application/octet-stream" },
            { "response_format", "json", "", "" },
            { "temperature", "0.0", "", "" },
        };
        auto res = cli.Post(partes.path, cabeceras(), items);
        if (!res) { error = "no responde " + url + " (" + httplib::to_string(res.error()) + ")"; return false; }
        if (res->status != 200) { error = url + " respondió " + std::to_string(res->status) + ": " + res->body.substr(0, 300); return false; }
        texto = sin_pensamiento(json::parse(res->body).value("text", std::string()));
        if (texto.empty()) { error = "el audio no tiene voz reconocible"; return false; }
        return true;
    } catch (const std::exception & e) {
        error = std::string("oído: ") + e.what();
        return false;
    }
}

// ---------------------------------------------------------------------------
// ffmpeg (opcional): convertir formatos y sacar cuadros/audio de un video
// ---------------------------------------------------------------------------

struct dir_temporal {
    std::filesystem::path p;
    dir_temporal() {
        const std::string base = nai_lector::ruta_temporal("d");
        p = std::filesystem::path(base);
        std::error_code ec;
        std::filesystem::create_directories(p, ec);
        std::filesystem::permissions(p, std::filesystem::perms::owner_all, std::filesystem::perm_options::replace, ec);
    }
    ~dir_temporal() { std::error_code ec; std::filesystem::remove_all(p, ec); }
    std::string escribir(const std::string & nombre, const std::string & bytes) const {
        const auto r = (p / nombre).string();
        std::ofstream f(r, std::ios::binary | std::ios::trunc);
        f.write(bytes.data(), (std::streamsize) bytes.size());
        return r;
    }
    static std::string leer(const std::string & r) {
        std::ifstream f(r, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    }
};

static std::string ext_mime(const std::string & mime) {
    static const std::map<std::string, std::string> m = {
        { "image/png", "png" }, { "image/jpeg", "jpg" }, { "image/gif", "gif" }, { "image/webp", "webp" }, { "image/bmp", "bmp" },
        { "audio/wav", "wav" }, { "audio/mpeg", "mp3" }, { "audio/flac", "flac" }, { "audio/ogg", "ogg" }, { "audio/mp4", "m4a" },
        { "audio/webm", "webm" }, { "video/mp4", "mp4" }, { "video/quicktime", "mov" }, { "video/webm", "webm" }, { "video/x-msvideo", "avi" },
    };
    auto it = m.find(mime);
    return it == m.end() ? "bin" : it->second;
}

// audio en WAV 16 kHz mono (lo que mejor leen los modelos de audio y whisper)
static bool a_wav(const std::string & bytes, const std::string & mime, std::string & wav, std::string & error) {
    dir_temporal d;
    const std::string in  = d.escribir("in." + ext_mime(mime), bytes);
    const std::string out = (d.p / "a.wav").string();
    std::string salida;
    if (!nai_lector::ejecutar({ "ffmpeg", "-v", "error", "-nostdin", "-y", "-i", in, "-vn", "-ac", "1", "-ar", "16000", "-t", "1800", out }, salida, error)) {
        error = "para este formato de audio hace falta ffmpeg (" + error + ")";
        return false;
    }
    wav = dir_temporal::leer(out);
    return !wav.empty();
}

// ---------------------------------------------------------------------------
// Ver, oír, video
// ---------------------------------------------------------------------------

static const char * SISTEMA_OJOS =
    "Eres los ojos de NAI. Describe en español, con precisión y sin inventar, lo que se ve: objetos, lugar, "
    "acciones, colores, cantidades y cualquier texto visible (transcríbelo tal cual). No identifiques a personas "
    "reales por su nombre. Si algo no se distingue, dilo. Responde solo con la descripción.";

bool ver(const std::string & bytes0, const std::string & mime0, const std::string & pedido, std::string & texto, std::string & error) {
    const config & c = actual();
    if (c.ojos.empty()) { error = "NAI no tiene ojos: arranca el servidor con un modelo de visión (--mmproj) o con --nai-ojos URL"; return false; }
    std::string bytes = bytes0, mime = mime0;
    if (mime != "image/png" && mime != "image/jpeg" && mime != "image/gif" && mime != "image/bmp") {   // webp y otros: a PNG si hay ffmpeg
        dir_temporal d;
        const std::string in = d.escribir("in." + ext_mime(mime), bytes), out = (d.p / "o.png").string();
        std::string s, e;
        if (nai_lector::ejecutar({ "ffmpeg", "-v", "error", "-nostdin", "-y", "-i", in, "-frames:v", "1", out }, s, e)) {
            bytes = dir_temporal::leer(out);
            mime  = "image/png";
        }
    }
    const std::string p = pedido.empty() ? "Describe esta imagen." : "Describe esta imagen. El usuario dijo: «" + pedido.substr(0, 600) + "»";
    json contenido = json::array({
        { { "type", "text" }, { "text", p } },
        { { "type", "image_url" }, { "image_url", { { "url", "data:" + mime + ";base64," + base64::encode(bytes) } } } },
    });
    return pedir_chat(c.ojos, contenido, SISTEMA_OJOS, 600, texto, error);
}

bool oir(const std::string & bytes0, const std::string & mime0, std::string & texto, std::string & error) {
    const config & c = actual();
    if (c.oido.empty()) { error = "NAI no tiene oído: arranca el servidor con un modelo de audio (--mmproj) o con --nai-oido URL"; return false; }
    std::string bytes = bytes0, mime = mime0;
    if (mime != "audio/wav" && mime != "audio/mpeg" && mime != "audio/flac") {
        std::string wav;
        if (!a_wav(bytes, mime, wav, error)) return false;
        bytes = std::move(wav);
        mime  = "audio/wav";
    }
    const std::string ruta = common_http_parse_url(c.oido).path;
    if (ruta.size() >= 10 && ruta.compare(ruta.size() - 10, 10, "/inference") == 0) {
        return pedir_whisper(c.oido, bytes, "audio." + ext_mime(mime), texto, error);
    }
    json contenido = json::array({
        { { "type", "input_audio" }, { "input_audio", { { "data", base64::encode(bytes) }, { "format", mime == "audio/mpeg" ? "mp3" : ext_mime(mime) } } } },
        { { "type", "text" }, { "text", "Transcribe este audio palabra por palabra, en su idioma original." } },
    });
    return pedir_chat(c.oido, contenido, "", 2000, texto, error);
}

bool ver_video(const std::string & bytes, const std::string & mime, const std::string & pedido, std::string & texto, std::string & error) {
    const config & c = actual();
    if (c.ojos.empty() && c.oido.empty()) { error = "NAI no tiene ojos ni oído para el video (--mmproj, --nai-ojos o --nai-oido)"; return false; }
    dir_temporal d;
    const std::string in = d.escribir("in." + ext_mime(mime), bytes);
    std::string s, e;
    double dur = 0;
    if (nai_lector::ejecutar({ "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", in }, s, e)) {
        try { dur = std::stod(s); } catch (...) { dur = 0; }
    } else if (e.find("no se encontró") != std::string::npos) {
        error = "para ver videos hace falta ffmpeg (en Mac: brew install ffmpeg)";
        return false;
    }
    std::string vista, oido_txt, err_v, err_o;
    if (!c.ojos.empty()) {
        char fps[32];
        snprintf(fps, sizeof(fps), "%.5f", dur > 0 ? std::max(0.001, c.cuadros / dur) : 1.0);
        if (nai_lector::ejecutar({ "ffmpeg", "-v", "error", "-nostdin", "-y", "-i", in, "-vf", std::string("fps=") + fps + ",scale=512:-2",
                                   "-frames:v", std::to_string(c.cuadros), (d.p / "c%02d.jpg").string() }, s, err_v)) {
            json contenido = json::array();
            contenido.push_back({ { "type", "text" }, { "text", "Estos son " + std::to_string(c.cuadros) + " cuadros de un video, en orden" +
                                  (dur > 0 ? " (dura " + std::to_string((int) (dur + 0.5)) + " s)" : std::string()) +
                                  ". Describe qué pasa en el video." + (pedido.empty() ? "" : " El usuario dijo: «" + pedido.substr(0, 600) + "»") } });
            for (int i = 1; i <= c.cuadros; ++i) {
                char n[16];
                snprintf(n, sizeof(n), "c%02d.jpg", i);
                const std::string cu = dir_temporal::leer((d.p / n).string());
                if (cu.empty()) break;
                contenido.push_back({ { "type", "image_url" }, { "image_url", { { "url", "data:image/jpeg;base64," + base64::encode(cu) } } } });
            }
            if (contenido.size() > 1) pedir_chat(c.ojos, contenido, SISTEMA_OJOS, 800, vista, err_v);
        }
    }
    if (!c.oido.empty()) {
        const std::string wav = (d.p / "a.wav").string();
        if (nai_lector::ejecutar({ "ffmpeg", "-v", "error", "-nostdin", "-y", "-i", in, "-vn", "-ac", "1", "-ar", "16000", "-t", "1800", wav }, s, err_o)) {
            oir(dir_temporal::leer(wav), "audio/wav", oido_txt, err_o);
        } else {
            err_o = "el video no tiene audio";
        }
    }
    if (vista.empty() && oido_txt.empty()) {
        error = !err_v.empty() ? err_v : (!err_o.empty() ? err_o : "no se pudo ver ni oír el video");
        return false;
    }
    texto = "Video" + (dur > 0 ? " de " + std::to_string((int) (dur + 0.5)) + " s" : std::string()) + ".";
    if (!vista.empty())    texto += "\nLo que se ve: " + vista;
    if (!oido_txt.empty()) texto += "\nLo que se oye: " + oido_txt;
    return true;
}

// los mismos medios se reenvían en cada turno (la interfaz manda el historial): se describen una vez
static std::mutex                         mtx_cache;
static std::map<std::string, std::string> cache;

bool describir(const std::string & tipo, const std::string & bytes, const std::string & mime, const std::string & pedido,
               std::string & texto, std::string & error) {
    const std::string clave = nai_cifrado::sha256_de(tipo + '\0' + pedido + '\0' + bytes);
    {
        std::lock_guard<std::mutex> lk(mtx_cache);
        auto it = cache.find(clave);
        if (it != cache.end()) { texto = it->second; return true; }
    }
    bool ok = false;
    if (tipo == "imagen")     ok = ver(bytes, mime, pedido, texto, error);
    else if (tipo == "audio") ok = oir(bytes, mime, texto, error);
    else if (tipo == "video") ok = ver_video(bytes, mime, pedido, texto, error);
    else error = "no es imagen, audio ni video";
    if (ok) {
        std::lock_guard<std::mutex> lk(mtx_cache);
        if (cache.size() >= 128) cache.clear();
        cache[clave] = texto;
    }
    return ok;
}

} // namespace nai_sentidos
