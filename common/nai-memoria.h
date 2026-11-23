// [NAI] Memoria reconstructiva por regiones (v3).
//
// Los pesos del modelo son la "corteza": el conocimiento general que ya está ahí y no cambia.
// Cada sesión tiene un "hipocampo": una memoria de TAMAÑO FIJO, organizada por regiones
// temáticas, que guarda solo lo que el modelo no podría adivinar.
//
//   EXACTO -> el texto codificado con el propio modelo (rANS): solo se guarda la sorpresa.
//             Se reconstruye idéntico recorriendo el mismo camino con el mismo modelo.
//   IDEA   -> una frase corta con la idea esencial, escrita por el modelo y codificada igual.
//   PISTA  -> las palabras más sorprendentes, para "volver a llegar" al recuerdo.
//
// Cuando la memoria se llena, lo menos importante baja de nivel (olvidar sin borrar).
// Importancia = peso de la región x refuerzo por uso + recencia. Las regiones prioritarias
// (peso 3) ceden al final. Cada recuerdo lleva claves (hash de palabras) para encontrarlo
// sin reconstruirlo: la sesión funciona como un GPS hacia sus recuerdos.
//
// Hay dos memorias: la de cada sesión (caduca con el TTL) y una biblioteca compartida
// (lo que se aprende de documentos y URLs; no caduca).
//
// La conversación ya no se acumula en el KV cache: el contexto solo lleva el mensaje
// actual, el turno anterior y los recuerdos relevantes reconstruidos para esa pregunta.
#pragma once

#include "llama.h"
#include "nai-cifrado.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace nai_mem {

struct region {
    int         peso = 1;     // 1 = general, 3 = prioritario (se conserva más)
    std::string nombre;
    std::string semilla;      // descripción del tema
};

std::vector<region> regiones_por_defecto();
std::vector<region> leer_regiones(const std::string & ruta);  // líneas "peso|tema|semilla"

enum nivel : uint8_t { EXACTO = 0, IDEA = 1, PISTA = 2 };
enum rol   : uint8_t { DOCUMENTO = 0, USUARIO = 1, ASISTENTE = 2 };
const char * nombre_nivel(int n);
const char * nombre_rol(int r);

// Texto codificado. Dos métodos:
//   tokens (portable, por defecto): IDs de token del vocabulario en enteros de longitud variable.
//          Sin números flotantes: se lee idéntico en cualquier máquina, backend o versión, y
//          decodificar no requiere pasar por el modelo (es una consulta de tabla).
//   modelo (compacto): rANS de 64 bits con las probabilidades del modelo (frecuencias de 24 bits).
//          Ocupa menos, pero solo se lee con el MISMO entorno (modelo, backend, -ngl, hilos, versión).
struct codigo {
    uint8_t               n_tokens = 0;
    bool                  tokens   = false;  // true = diccionario de tokens (portable)
    std::vector<uint32_t> palabras;
    size_t bytes() const { return 2 + 4 * palabras.size(); }
};

// Diccionario de tokens: varint de 7 bits por byte, empacado en palabras de 32 bits (little-endian).
// desempacar valida longitud e IDs < n_vocab: un recuerdo dañado se rechaza, nunca se lee fuera de rango.
std::vector<uint32_t> empacar_tokens(const std::vector<llama_token> & t);
bool                  desempacar_tokens(const codigo & c, int n_vocab, std::vector<llama_token> & t);

struct recuerdo {
    uint8_t               region = 0;
    uint8_t               nivel  = EXACTO;
    uint8_t               rol    = DOCUMENTO;
    uint8_t               usos   = 0;
    uint8_t               check  = 0;   // suma de verificación del texto del nivel actual
    uint16_t              dia    = 0;   // último uso, en días desde 2026-01-01
    std::vector<uint16_t> claves;       // hash de las palabras más sorprendentes
    codigo                cod;
    std::string           cache;        // texto reconstruido (solo en RAM, no se guarda)
    bool                  danado = false;

    // región, nivel+rol, usos, check, día (2), n_claves + claves + código
    size_t bytes() const { return 7 + 2 * claves.size() + cod.bytes(); }
};

struct memoria {
    std::vector<recuerdo> recuerdos;
    // Memoria de trabajo: los últimos turnos, literales, en anillo dentro de la cadena
    // (separados por SEP_TURNO; un .mem anterior sin separador se lee como un solo turno).
    std::string           previo_usuario;
    std::string           previo_asistente;
    size_t                presupuesto = 16384;
    bool                  sucia       = false;
    int64_t               t_acceso    = 0;

    size_t bytes() const;
};

struct estadisticas {
    size_t n_exacto = 0, n_idea = 0, n_pista = 0, n_danados = 0;
    size_t bytes = 0, presupuesto = 0;
    size_t bytes_texto = 0;      // lo que ocuparía el texto original de los recuerdos exactos
};

// Resultado de recordar: lo que se le da al modelo antes de responder.
struct contexto_recordado {
    std::string bloque;           // "Recuerdos: ..." para el mensaje de sistema
    std::vector<std::pair<std::string, std::string>> turnos;   // últimos turnos (usuario, asistente), del más viejo al más nuevo
    int         n_recuerdos = 0;
};

// Motor: un contexto propio del modelo para codificar, decodificar, clasificar y resumir.
// No es thread-safe; el servicio lo protege con un mutex.
class motor {
public:
    motor(llama_model * model, int n_threads, std::vector<region> regiones);
    ~motor();
    bool ok() const { return ctx != nullptr; }

    const std::vector<region> & regiones() const { return regs; }
    uint64_t huella() const { return huella_; }

    // Agrega un texto (se parte en trozos) en una región; region < 0 = clasificar con el modelo.
    // Devuelve el número de recuerdos agregados.
    int agregar(memoria & m, const std::string & texto, int region = -1, int rol = DOCUMENTO);

    // Reconstruye el texto del nivel actual de un recuerdo.
    const std::string & texto(recuerdo & r);

    // Busca los recuerdos relevantes para una consulta en una o varias memorias.
    // Refuerza los usados (usos++, día de hoy).
    std::string recordar(std::vector<memoria *> ms, const std::string & consulta, int max_recuerdos, size_t max_chars, int * n_usados);

    // Ajusta la memoria a su presupuesto bajando de nivel lo menos importante.
    void ajustar(memoria & m);

    int clasificar(const std::string & texto);
    std::string generar(const std::string & instruccion, int n_max);

    estadisticas medir(const memoria & m) const;
    double importancia(const recuerdo & r) const;

    // Con clave activa los archivos se cifran (ChaCha20-Poly1305); los archivos sin cifrar se siguen leyendo.
    void usar_clave(const nai_cifrado::clave & c) { cla = c; }
    // true = codificar con el modelo (compacto, ligado al entorno); false = diccionario de tokens (portable)
    void usar_codificacion_modelo(bool si) { con_modelo = si; }
    bool guardar(const memoria & m, const std::string & ruta) const;
    bool cargar(memoria & m, const std::string & ruta) const;

    // Utilidades públicas para pruebas
    bool codificar(const std::string & prefijo, const std::string & texto, codigo & out,
                   std::string * texto_tokens = nullptr, std::vector<std::pair<std::string, double>> * palabras = nullptr);
    bool decodificar(const std::string & prefijo, const codigo & c, std::string & out);
    std::string prefijo(int region, int nivel, int rol) const;

private:
    void bajar_nivel(recuerdo & r);
    std::vector<llama_token> tokenizar(const std::string & s, bool especiales) const;
    std::string detokenizar(const std::vector<llama_token> & t) const;
    std::string plantilla(const std::string & usuario) const;
    bool evaluar(const std::vector<llama_token> & t);

    llama_model *       model = nullptr;
    const llama_vocab * vocab = nullptr;
    llama_context *     ctx   = nullptr;
    int                 n_vocab = 0;
    std::vector<region> regs;
    uint64_t            huella_ = 0;
    nai_cifrado::clave  cla;
    bool                con_modelo = false;
    std::vector<uint32_t> cum;
};

// Partir un texto largo en trozos de hasta max_chars, respetando párrafos y oraciones.
std::vector<std::string> trocear(const std::string & texto, size_t max_chars = 600, size_t min_chars = 24);

// Claves: palabras normalizadas (minúsculas, sin acentos, raíz de 6 letras) -> hash de 16 bits.
std::vector<uint16_t> claves_de(const std::string & texto);
uint16_t clave_palabra(const std::string & palabra);   // 0 = palabra vacía o de relleno
uint16_t hoy();

// Servicio para el servidor: memorias por sesión + biblioteca, con un hilo de trabajo.
class servicio {
public:
    struct config {
        std::string directorio;                     // donde se guardan los .mem
        size_t      bytes_sesion     = 16384;
        size_t      bytes_biblioteca = 8u << 20;
        int64_t     ttl_seg          = 86400;
        int         n_threads        = 4;
        int         max_recuerdos    = 6;
        size_t      max_chars        = 2400;
        int         turnos           = 3;           // turnos recientes que se conservan literales
        std::string ruta_regiones;                  // vacío = regiones por defecto
        nai_cifrado::clave clave;                   // activa = cifrar los .mem
        bool        compacta         = false;       // true = codificar con el modelo (no portable)
    };

    servicio(llama_model * model, const config & cfg);
    ~servicio();
    bool ok() const { return mot && mot->ok(); }
    int  turnos() const { return cfg.turnos; }

    // Antes de responder (sincrónico): recuerdos relevantes + turno anterior.
    contexto_recordado preparar(uint64_t sesion, const std::string & consulta);

    // Después de responder (asíncrono): guarda el turno en la memoria de la sesión.
    void registrar_turno(uint64_t sesion, const std::string & usuario, const std::string & asistente);

    // Aprender un texto en la biblioteca (asíncrono). Devuelve cuántos trozos se encolaron.
    // region: nombre de la región si la fuente ya la conoce (vacío = el modelo clasifica cada trozo).
    int aprender(const std::string & texto, const std::string & fuente, const std::string & region = "");
    int indice_region(const std::string & nombre) const;  // -1 si no existe

    std::string estado_json();  // estado de la biblioteca y del trabajo pendiente (JSON)

    // Regla anti-congelamiento: el servidor avisa cuándo empieza y termina cada petición. El trabajo
    // de fondo (codificar turnos, aprender documentos) espera mientras haya alguna en curso, para no
    // competir por CPU/GPU con la respuesta (con un tope de espera para que la cola no se atore).
    void peticion(bool empieza) { activas.fetch_add(empieza ? 1 : -1); }

private:
    memoria & sesion_mem(uint64_t sesion);   // requiere mtx
    std::string ruta_sesion(uint64_t sesion) const;
    void purgar();                           // requiere mtx
    void encolar(std::function<void()> f);
    void bucle();

    config                           cfg;
    std::unique_ptr<motor>           mot;
    std::mutex                       mtx;     // protege motor y memorias
    std::map<uint64_t, memoria>      sesiones;
    memoria                          biblioteca;
    int64_t                          t_purga = 0;

    std::mutex                         mtx_cola;
    std::condition_variable            cv;
    std::deque<std::function<void()>>  cola;
    bool                               salir = false;
    std::thread                        hilo;
    size_t                             pendientes_aprender = 0;
    std::atomic<int>                   activas{0};
    size_t                             aprendidos = 0;
};

} // namespace nai_mem
