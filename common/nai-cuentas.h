// [NAI] Cuentas y acceso: NAI es UNA entidad que se consulta desde varios dispositivos.
//
//   dueño     acceso completo: enseña (conversación y documentos), administra cuentas, entrena,
//             manda código y comandos. Ve también lo privado.
//   invitado  amigos y parientes: solo consultan el conocimiento general de la entidad. No ven lo
//             privado del dueño y no pueden modificar nada: ni enseñar, ni inyectar código, datos o
//             instrucciones, ni entrenar. Su plática se recuerda en su propia sesión (aislada, caduca).
//
// Contraseñas: PBKDF2-HMAC-SHA256 con sal aleatoria de 16 bytes (210 000 iteraciones, guardadas por
// cuenta para poder subirlas). Tokens: 256 bits aleatorios; en disco solo su SHA-256. Caducan
// (30 días por defecto, se renuevan solos con el uso) y se revocan al cerrar sesión o al borrar la
// cuenta. Tras 5 intentos fallidos, la cuenta y la dirección se bloquean (1 min, luego el doble,
// hasta 15 min). El archivo de cuentas se cifra con la clave de --nai-cifrado si está activa.
#pragma once

#include "nai-cifrado.h"

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace nai_cuentas {

enum rol : uint8_t { DUENO = 0, INVITADO = 1 };
const char * nombre_rol(int r);

struct identidad {
    bool        ok = false;
    std::string usuario;
    int         rol = INVITADO;
    std::string dispositivo;
    int64_t     expira = 0;
    bool es_dueno() const { return ok && rol == DUENO; }
};

struct resultado_login {
    bool        ok = false;
    std::string token;
    int64_t     expira = 0;
    int         rol = INVITADO;
    std::string error;
    int         espera_seg = 0;   // > 0: bloqueado por intentos fallidos
};

class gestor {
public:
    // directorio: donde vive cuentas.dat. clave: la de --nai-cifrado (puede estar inactiva).
    gestor(const std::string & directorio, const nai_cifrado::clave & clave, int dias_token = 30, uint32_t iteraciones = 210000);

    bool hay_cuentas();
    // Primera vez: crea al dueño. Solo funciona si no existe ninguna cuenta.
    bool configurar(const std::string & usuario, const std::string & contrasena, std::string & error);
    resultado_login login(const std::string & usuario, const std::string & contrasena,
                          const std::string & dispositivo, const std::string & origen);
    identidad validar(const std::string & token);   // rápido: SHA-256 + búsqueda; renueva si falta poco
    bool      cerrar(const std::string & token);
    // Solo el dueño (lo verifica quien llama con identidad::es_dueno)
    bool crear(const std::string & usuario, const std::string & contrasena, int rol, std::string & error);
    bool borrar(const std::string & usuario, std::string & error);   // revoca sus tokens; el dueño no se borra
    bool cambiar_contrasena(const std::string & usuario, const std::string & actual, const std::string & nueva, std::string & error);
    std::string listar_json();

    static bool usuario_valido(const std::string & u);     // 3-32: a-z 0-9 . _ -
    static bool contrasena_valida(const std::string & c, std::string & error);   // mínimo 10 caracteres

private:
    struct cuenta {
        std::string usuario;
        uint8_t     rol = INVITADO;
        uint32_t    iter = 0;
        std::string sal, hash;   // binarios
        int64_t     creada = 0;
    };
    struct token {
        std::string usuario;
        uint8_t     rol = INVITADO;
        int64_t     expira = 0;
        std::string dispositivo;
    };
    struct intentos {
        int     fallos = 0;
        int64_t hasta = 0;
        int     castigo = 60;
    };

    std::string derivar(const std::string & contrasena, const std::string & sal, uint32_t iter) const;
    bool cargar();
    bool guardar();   // requiere mtx
    void podar();     // requiere mtx: tokens vencidos

    std::string                       ruta;
    nai_cifrado::clave                cla;
    int64_t                           ttl;
    uint32_t                          iteraciones;
    std::mutex                        mtx;
    std::map<std::string, cuenta>     cuentas;
    std::map<std::string, token>      tokens;    // SHA-256(token) -> datos
    std::map<std::string, intentos>   fallos;    // "u:usuario" y "o:origen"
};

// Gestor activo (lo usa el middleware HTTP para validar tokens antes de llegar a las rutas).
gestor * activo();
void     activar(gestor * g);

} // namespace nai_cuentas
