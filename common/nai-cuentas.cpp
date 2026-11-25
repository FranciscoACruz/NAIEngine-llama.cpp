// [NAI] Cuentas y acceso. Ver nai-cuentas.h.
#include "nai-cuentas.h"

#include <algorithm>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace nai_cuentas {

static std::atomic<gestor *> g_activo{ nullptr };
gestor * activo() { return g_activo.load(); }
void activar(gestor * g) { g_activo.store(g); }

const char * nombre_rol(int r) { return r == DUENO ? "dueno" : "invitado"; }

static int64_t ahora() { return (int64_t) std::time(nullptr); }

static std::string json_esc(const std::string & s) {
    std::string r;
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { r += '\\'; r += (char) c; }
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); r += b; }
        else r += (char) c;
    }
    return r;
}

gestor::gestor(const std::string & directorio, const nai_cifrado::clave & clave, int dias_token, uint32_t iter)
    : ruta((std::filesystem::path(directorio) / "cuentas.dat").string()), cla(clave),
      ttl((int64_t) std::max(1, dias_token) * 86400), iteraciones(std::max<uint32_t>(10000, iter)) {
    std::error_code ec;
    std::filesystem::create_directories(directorio, ec);
    cargar();
}

bool gestor::usuario_valido(const std::string & u) {
    if (u.size() < 3 || u.size() > 32) return false;
    return std::all_of(u.begin(), u.end(), [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
    });
}

bool gestor::contrasena_valida(const std::string & c, std::string & error) {
    if (c.size() < 10) { error = "la contraseña debe tener al menos 10 caracteres"; return false; }
    if (c.size() > 256) { error = "la contraseña es demasiado larga"; return false; }
    return true;
}

std::string gestor::derivar(const std::string & contrasena, const std::string & sal, uint32_t iter) const {
    uint8_t out[32];
    nai_cifrado::pbkdf2_sha256(contrasena, (const uint8_t *) sal.data(), sal.size(), iter, out, 32);
    return std::string((const char *) out, 32);
}

bool gestor::hay_cuentas() {
    std::lock_guard<std::mutex> lk(mtx);
    return !cuentas.empty();
}

bool gestor::configurar(const std::string & usuario, const std::string & contrasena, std::string & error) {
    if (!usuario_valido(usuario)) { error = "usuario inválido: 3 a 32 caracteres a-z, 0-9, punto, guion o guion bajo"; return false; }
    if (!contrasena_valida(contrasena, error)) return false;
    uint8_t s[16];
    nai_cifrado::aleatorio(s, 16);
    cuenta c;
    c.usuario = usuario; c.rol = DUENO; c.iter = iteraciones; c.creada = ahora();
    c.sal = std::string((const char *) s, 16);
    c.hash = derivar(contrasena, c.sal, c.iter);
    std::lock_guard<std::mutex> lk(mtx);
    if (!cuentas.empty()) { error = "NAI ya tiene dueño; inicia sesión"; return false; }
    cuentas[usuario] = c;
    if (!guardar()) { cuentas.clear(); error = "no se pudo guardar el archivo de cuentas"; return false; }
    return true;
}

resultado_login gestor::login(const std::string & usuario, const std::string & contrasena,
                              const std::string & dispositivo, const std::string & origen) {
    resultado_login r;
    const int64_t t = ahora();
    const std::string ku = "u:" + usuario, ko = "o:" + origen;
    cuenta c;
    bool existe = false;
    {
        std::lock_guard<std::mutex> lk(mtx);
        for (const std::string & k : { ku, ko }) {
            auto it = fallos.find(k);
            if (it != fallos.end() && it->second.hasta > t) {
                r.espera_seg = (int) (it->second.hasta - t);
                r.error = "demasiados intentos fallidos; espera " + std::to_string(r.espera_seg) + " s";
                return r;
            }
        }
        auto it = cuentas.find(usuario);
        if (it != cuentas.end()) { c = it->second; existe = true; }
    }
    // Se deriva aunque el usuario no exista (mismo tiempo): no revela qué usuarios hay.
    static const std::string sal_falsa(16, '\x5a');
    const std::string h = derivar(contrasena, existe ? c.sal : sal_falsa, existe ? c.iter : iteraciones);
    std::lock_guard<std::mutex> lk(mtx);
    if (!existe || !nai_cifrado::iguales_tiempo_constante(h, c.hash)) {
        // 5 fallos por usuario; 20 por dirección (detrás de un proxy o en casa todos comparten IP:
        // un invitado que se equivoca no debe dejar fuera a toda la familia)
        for (const std::string & k : { ku, ko }) {
            intentos & in = fallos[k];
            if (++in.fallos >= (k[0] == 'u' ? 5 : 20)) {
                in.hasta = t + in.castigo;
                in.castigo = std::min(in.castigo * 2, 900);
                in.fallos = 0;
            }
        }
        r.error = "usuario o contraseña incorrectos";
        return r;
    }
    fallos.erase(ku);
    fallos.erase(ko);
    uint8_t b[32];
    nai_cifrado::aleatorio(b, 32);
    r.token = "nai_" + nai_cifrado::base64url(b, 32);
    r.expira = t + ttl;
    r.rol = c.rol;
    r.ok = true;
    token tk;
    tk.usuario = usuario; tk.rol = c.rol; tk.expira = r.expira;
    tk.dispositivo = dispositivo.substr(0, 64);
    tokens[nai_cifrado::sha256_de(r.token)] = tk;
    podar();
    guardar();
    return r;
}

identidad gestor::validar(const std::string & tok) {
    identidad id;
    if (tok.size() < 10 || tok.rfind("nai_", 0) != 0) return id;
    const std::string h = nai_cifrado::sha256_de(tok);
    const int64_t t = ahora();
    std::lock_guard<std::mutex> lk(mtx);
    auto it = tokens.find(h);
    if (it == tokens.end()) return id;
    if (it->second.expira <= t) { tokens.erase(it); guardar(); return id; }
    auto ic = cuentas.find(it->second.usuario);
    if (ic == cuentas.end()) { tokens.erase(it); guardar(); return id; }
    // renovación deslizante: si ya pasó la mitad de su vida, se extiende (se guarda sin bloquear mucho)
    if (it->second.expira - t < ttl / 2) { it->second.expira = t + ttl; guardar(); }
    id.ok = true;
    id.usuario = it->second.usuario;
    id.rol = ic->second.rol;           // el rol vigente de la cuenta, no el del momento del login
    id.dispositivo = it->second.dispositivo;
    id.expira = it->second.expira;
    return id;
}

bool gestor::cerrar(const std::string & tok) {
    std::lock_guard<std::mutex> lk(mtx);
    const bool ok = tokens.erase(nai_cifrado::sha256_de(tok)) > 0;
    if (ok) guardar();
    return ok;
}

bool gestor::crear(const std::string & usuario, const std::string & contrasena, int rol_, std::string & error) {
    if (!usuario_valido(usuario)) { error = "usuario inválido: 3 a 32 caracteres a-z, 0-9, punto, guion o guion bajo"; return false; }
    if (!contrasena_valida(contrasena, error)) return false;
    if (rol_ != INVITADO) { error = "solo se pueden crear cuentas de invitado (NAI tiene un solo dueño)"; return false; }
    uint8_t s[16];
    nai_cifrado::aleatorio(s, 16);
    cuenta c;
    c.usuario = usuario; c.rol = INVITADO; c.iter = iteraciones; c.creada = ahora();
    c.sal = std::string((const char *) s, 16);
    c.hash = derivar(contrasena, c.sal, c.iter);
    std::lock_guard<std::mutex> lk(mtx);
    if (cuentas.count(usuario)) { error = "ese usuario ya existe"; return false; }
    cuentas[usuario] = c;
    if (!guardar()) { cuentas.erase(usuario); error = "no se pudo guardar el archivo de cuentas"; return false; }
    return true;
}

bool gestor::borrar(const std::string & usuario, std::string & error) {
    std::lock_guard<std::mutex> lk(mtx);
    auto it = cuentas.find(usuario);
    if (it == cuentas.end()) { error = "no existe ese usuario"; return false; }
    if (it->second.rol == DUENO) { error = "la cuenta del dueño no se puede borrar"; return false; }
    cuentas.erase(it);
    for (auto t = tokens.begin(); t != tokens.end();) t = t->second.usuario == usuario ? tokens.erase(t) : std::next(t);
    return guardar();
}

bool gestor::cambiar_contrasena(const std::string & usuario, const std::string & actual, const std::string & nueva, std::string & error) {
    if (!contrasena_valida(nueva, error)) return false;
    cuenta c;
    {
        std::lock_guard<std::mutex> lk(mtx);
        auto it = cuentas.find(usuario);
        if (it == cuentas.end()) { error = "no existe ese usuario"; return false; }
        c = it->second;
    }
    if (!nai_cifrado::iguales_tiempo_constante(derivar(actual, c.sal, c.iter), c.hash)) { error = "la contraseña actual no coincide"; return false; }
    uint8_t s[16];
    nai_cifrado::aleatorio(s, 16);
    c.sal = std::string((const char *) s, 16);
    c.iter = iteraciones;
    c.hash = derivar(nueva, c.sal, c.iter);
    std::lock_guard<std::mutex> lk(mtx);
    cuentas[usuario] = c;
    // al cambiar la contraseña se cierran todas sus sesiones
    for (auto t = tokens.begin(); t != tokens.end();) t = t->second.usuario == usuario ? tokens.erase(t) : std::next(t);
    return guardar();
}

std::string gestor::listar_json() {
    std::lock_guard<std::mutex> lk(mtx);
    podar();
    std::string r = "[";
    for (auto & kv : cuentas) {
        size_t n_tok = 0;
        for (auto & t : tokens) n_tok += t.second.usuario == kv.first;
        if (r.size() > 1) r += ",";
        r += "{\"usuario\":\"" + json_esc(kv.first) + "\",\"rol\":\"" + nombre_rol(kv.second.rol) +
             "\",\"creada\":" + std::to_string(kv.second.creada) + ",\"sesiones_abiertas\":" + std::to_string(n_tok) + "}";
    }
    return r + "]";
}

void gestor::podar() {
    const int64_t t = ahora();
    for (auto it = tokens.begin(); it != tokens.end();) it = it->second.expira <= t ? tokens.erase(it) : std::next(it);
    for (auto it = fallos.begin(); it != fallos.end();) it = (it->second.hasta <= t && it->second.fallos == 0) ? fallos.erase(it) : std::next(it);
}

// Formato (texto, una entrada por línea, campos separados por tabuladores):
//   c  usuario  rol  iteraciones  sal_hex  hash_hex  creada
//   t  sha256(token)_hex  usuario  rol  expira  dispositivo_hex
bool gestor::guardar() {
    std::ostringstream o;
    o << "NAICUENTAS1\n";
    for (auto & kv : cuentas) {
        const cuenta & c = kv.second;
        o << "c\t" << c.usuario << "\t" << (int) c.rol << "\t" << c.iter << "\t" << nai_cifrado::a_hex(c.sal) << "\t"
          << nai_cifrado::a_hex(c.hash) << "\t" << c.creada << "\n";
    }
    for (auto & kv : tokens) {
        const token & t = kv.second;
        o << "t\t" << nai_cifrado::a_hex(kv.first) << "\t" << t.usuario << "\t" << (int) t.rol << "\t" << t.expira << "\t"
          << nai_cifrado::a_hex(t.dispositivo) << "\n";
    }
    std::string b = o.str();
    if (cla.activa) b = nai_cifrado::cifrar(cla, b, "cuentas.dat");
    const std::string tmp = ruta + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f || !f.write(b.data(), (std::streamsize) b.size())) return false;
    }
    std::error_code ec;
    std::filesystem::permissions(tmp, std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
                                 std::filesystem::perm_options::replace, ec);
    std::filesystem::rename(tmp, ruta, ec);
    return !ec;
}

bool gestor::cargar() {
    std::ifstream f(ruta, std::ios::binary);
    if (!f) return false;
    std::string b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();   // en Windows un archivo abierto no se puede renombrar (apartarlo abajo fallaría)
    if (nai_cifrado::es_cifrado(b)) {
        std::string plano;
        if (!cla.activa || !nai_cifrado::descifrar(cla, b, "cuentas.dat", plano)) {
            // no se sobrescribe: se aparta para no perder las cuentas
            std::error_code ec;
            std::filesystem::rename(ruta, ruta + ".ilegible-" + std::to_string(ahora()), ec);
            fprintf(stderr, "[NAI] cuentas: no se pudo descifrar %s (otra clave o sin --nai-cifrado); %s y se empieza sin cuentas\n",
                    ruta.c_str(), ec ? "NO se pudo apartar" : "se apartó");
            return false;
        }
        b = std::move(plano);
    }
    std::istringstream in(b);
    std::string l;
    if (!std::getline(in, l) || l != "NAICUENTAS1") return false;
    while (std::getline(in, l)) {
        std::vector<std::string> c;
        std::string campo;
        std::istringstream ls(l);
        while (std::getline(ls, campo, '\t')) c.push_back(campo);
        if (c.size() == 7 && c[0] == "c") {
            cuenta x;
            x.usuario = c[1]; x.rol = (uint8_t) std::stoi(c[2]); x.iter = (uint32_t) std::stoul(c[3]);
            x.sal = nai_cifrado::de_hex(c[4]); x.hash = nai_cifrado::de_hex(c[5]); x.creada = std::stoll(c[6]);
            if (usuario_valido(x.usuario) && x.hash.size() == 32) cuentas[x.usuario] = x;
        } else if ((c.size() == 6 || c.size() == 5) && c[0] == "t") {
            token t;
            t.usuario = c[2]; t.rol = (uint8_t) std::stoi(c[3]); t.expira = std::stoll(c[4]);
            if (c.size() == 6) t.dispositivo = nai_cifrado::de_hex(c[5]);
            tokens[nai_cifrado::de_hex(c[1])] = t;
        }
    }
    std::lock_guard<std::mutex> lk(mtx);
    podar();
    return true;
}

} // namespace nai_cuentas
