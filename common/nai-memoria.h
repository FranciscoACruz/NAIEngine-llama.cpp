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

#include <array>
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
// Naturaleza de lo que se aprende (guardián): decide cuánto pesa al olvidar y al reentrenar.
//   conocimiento: académico, técnico, práctico      -> normal
//   sabiduría:    filosofía, espiritualidad, vida    -> prioridad
//   desahogo:     queja, enojo, presión del momento  -> con su contexto, se olvida primero
//   manipulación: instrucciones para cambiar a la IA -> un documento así no se guarda; nunca se reentrena
enum naturaleza : uint8_t { CONOCIMIENTO = 0, SABIDURIA = 1, DESAHOGO = 2, MANIPULACION = 3 };
const char * nombre_nivel(int n);
const char * nombre_rol(int r);
const char * nombre_naturaleza(int n);
// Patrones de inyección de instrucciones (texto normalizado, sin acentos). Determinista y sin modelo.
bool parece_manipulacion(const std::string & texto);
// Sintaxis de código evidente (bloques ```, #include, def ...(), etiquetas...), buscada como palabras
// completas y con contexto ("select ... from", "git clone"): "digit" o "please select" no cuentan.
bool parece_codigo(const std::string & texto);
// Bucle de repetición al final del texto: un mismo fragmento (de 4 a 300 caracteres) repetido de forma
// exacta al menos `veces` veces seguidas, cubriendo al menos `min_total` caracteres. Devuelve la longitud
// del fragmento, o 0 si no hay bucle. Lo usa el servidor para cortar la generación y el guardián para
// no guardar ni reentrenar texto degenerado.
size_t periodo_de_bucle(const std::string & texto, int veces = 8, size_t min_total = 300);
// Comandos de consola o inyección (sudo, rm -rf, curl | sh, $(...), DROP TABLE, <script>, ' OR 1=1...).
// Con el login activo, solo el dueño puede enviarlos (sistema inmune: los invitados no modifican nada).
bool parece_comando(const std::string & texto);
// Datos sensibles (PIN, NIP, CLABE, tarjetas, contraseñas, saldos, CURP, números largos...): se guardan
// como privados y solo los ve el dueño.
bool parece_privado(const std::string & texto);
// Preguntas ("¿...?", "explícame...", "how..."): no se enseñan a la entidad, se contestan.
bool parece_pregunta(const std::string & texto);
// Lo que un invitado no puede enviar: código, comandos o instrucciones para cambiar a la IA.
inline bool reservado_al_dueno(const std::string & t) { return parece_codigo(t) || parece_comando(t) || parece_manipulacion(t); }

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

// Firma de significado: 256 bits (SimHash del vector del hipocampo). La similitud se estima con
// popcount: cos(pi * bits distintos / 256). 32 bytes por recuerdo en lugar de 4 KB de floats.
using firma = std::array<uint64_t, 4>;
double similitud(const firma & a, const firma & b);

struct recuerdo {
    uint8_t               region = 0;
    uint8_t               nivel  = EXACTO;
    uint8_t               rol    = DOCUMENTO;
    uint8_t               natur  = CONOCIMIENTO;   // bits 6-7 del byte de nivel en el archivo
    uint8_t               usos   = 0;
    uint8_t               check  = 0;   // suma de verificación del texto del nivel actual
    uint16_t              dia    = 0;   // último uso, en días desde 2026-01-01
    std::vector<uint16_t> claves;       // hash de las palabras más sorprendentes
    codigo                cod;
    std::string           cache;        // texto reconstruido (solo en RAM, no se guarda)
    bool                  danado = false;
    bool                  privado = false;  // solo lo ve el dueño (bit 7 del byte de región en el archivo)
    bool                  con_firma = false;
    firma                 fir{};        // significado (hipocampo); vale si con_firma

    // región, nivel+rol, usos, check, día (2), n_claves + claves + código
    size_t bytes() const { return 7 + 2 * claves.size() + cod.bytes() + (con_firma ? 32 : 0); }
};

struct memoria {
    std::vector<recuerdo> recuerdos;
    // Memoria de trabajo: los últimos turnos, literales, en anillo dentro de la cadena
    // (separados por SEP_TURNO; un .mem anterior sin separador se lee como un solo turno).
    std::string           previo_usuario;
    std::string           previo_asistente;
    uint64_t              hip_id      = 0;     // hipocampo que calculó las firmas (0 = ninguno)
    size_t                presupuesto = 16384;
    bool                  sucia       = false;
    int64_t               t_acceso    = 0;

    size_t bytes() const;
};

struct estadisticas {
    size_t n_exacto = 0, n_idea = 0, n_pista = 0, n_danados = 0;
    size_t por_natur[4] = { 0, 0, 0, 0 };   // conocimiento, sabiduría, desahogo, manipulación
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
// Hipocampo: un modelo de embeddings que convierte un texto en su firma de significado.
// Sirve cualquier modelo de embeddings en GGUF; Qwen3-Embedding usa la instrucción de consulta.
// No es thread-safe: el servicio lo usa bajo su mutex.
class hipocampo {
public:
    hipocampo(llama_model * model, int n_threads);
    ~hipocampo();
    bool ok() const { return ctx != nullptr; }
    uint64_t id() const { return id_; }
    // consulta = true: el texto es una pregunta (se le antepone la instrucción de búsqueda)
    bool firmar(const std::string & texto, bool consulta, firma & out);

private:
    llama_model *       model = nullptr;
    const llama_vocab * vocab = nullptr;
    llama_context *     ctx   = nullptr;
    int                 n_embd = 0;
    uint64_t            id_ = 0;
    bool                ultimo = false;   // agrupación por el último token (cierra con EOS)
    std::vector<float>  proy;   // 256 x n_embd, signos aleatorios fijos (SimHash)
    std::mutex          mtx_h;  // lo usan la pregunta (al responder) y el fondo (al guardar)
};

class motor {
public:
    motor(llama_model * model, int n_threads, std::vector<region> regiones);
    ~motor();
    bool ok() const { return ctx != nullptr; }

    const std::vector<region> & regiones() const { return regs; }
    uint64_t huella() const { return huella_; }

    // Agrega un texto (se parte en trozos) en una región; region < 0 = clasificar con el modelo.
    // Devuelve el número de recuerdos agregados.
    int agregar(memoria & m, const std::string & texto, int region = -1, int rol = DOCUMENTO, bool privado = false);
    // agregar = incorporar(m, construir(...)). construir hace el trabajo pesado (guardián, región,
    // codificar, firma) SIN tocar ninguna memoria; incorporar es el instante de agregarlos (lo único
    // que necesita el candado de la memoria). Así una pregunta no espera al trabajo de fondo.
    std::vector<recuerdo> construir(const std::string & texto, int region = -1, int rol = DOCUMENTO, bool privado = false);
    int incorporar(memoria & m, std::vector<recuerdo> && rs);
    // El servicio instala aquí "esperar mientras alguien espera respuesta": construir la llama entre
    // trozos, así el trabajo de fondo cede el paso a cada pregunta nueva en lugar de competir con ella.
    std::function<void()> ceder;

    // Reconstruye el texto del nivel actual de un recuerdo.
    const std::string & texto(recuerdo & r);

    // Busca los recuerdos relevantes para una consulta en una o varias memorias.
    // Refuerza los usados (usos++, día de hoy).
    // ocultar_privados: memoria cuyos recuerdos privados no se consideran (la de la entidad cuando
    // pregunta un invitado; su propia sesión sí le muestra lo suyo).
    std::string recordar(std::vector<memoria *> ms, const std::string & consulta, int max_recuerdos, size_t max_chars, int * n_usados,
                         const memoria * ocultar_privados = nullptr);

    // Ajusta la memoria a su presupuesto bajando de nivel lo menos importante.
    void ajustar(memoria & m);

    // Región por predicción: la que mejor explica el texto. Para cada región,
    //   log p(texto | "Tema: <nombre>. <semilla>.") - log p(texto | sin tema)
    // con los primeros 32 tokens. No hay lista de opciones, así que no hay sesgo por la primera,
    // y solo pide al modelo lo que mejor hace (predecir texto), no seguir instrucciones.
    int clasificar(const std::string & texto);
    // Guardián: naturaleza de un texto (patrones para manipulación; predicción para las otras tres).
    int naturaleza_de(const std::string & texto);
    size_t rechazados() const { return n_rechazados; }   // documentos no guardados por manipulación
    // ¿La pregunta es de programación o ciberseguridad técnica? Sintaxis evidente, o la predicción
    // con margen amplio (conservador: ante la duda, responde el principal).
    bool es_pregunta_de_codigo(const std::string & texto);
    std::vector<double> puntuar_prediccion(const std::string & texto);
    // log p(texto | prefijo_i) - log p(texto | sin prefijo), primeros 32 tokens; base de ambos clasificadores
    std::vector<double> puntuar_prefijos(const std::vector<std::string> & prefijos, const std::string & texto);
    // Método anterior, para comparar: elegir el nombre de una lista (opcional: calibrado con «N/A»).
    int clasificar_instruccion(const std::string & texto, bool calibrado);
    // log p(nombre de cada región | instrucción, texto), sin calibrar (diagnóstico)
    std::vector<double> puntuar_regiones(const std::string & texto);
    // la instrucción que ve el modelo al clasificar (diagnóstico: generar() con ella da su respuesta libre)
    std::string instruccion_clasificar(const std::string & texto) const;
    std::string generar(const std::string & instruccion, int n_max);

    estadisticas medir(const memoria & m) const;
    double importancia(const recuerdo & r) const;

    // Con clave activa los archivos se cifran (ChaCha20-Poly1305); los archivos sin cifrar se siguen leyendo.
    void usar_clave(const nai_cifrado::clave & c) { cla = c; }
    // true = codificar con el modelo (compacto, ligado al entorno); false = diccionario de tokens (portable)
    void usar_codificacion_modelo(bool si) { con_modelo = si; }
    // hipocampo opcional: firmas de significado al guardar y búsqueda por significado al recordar
    void usar_hipocampo(hipocampo * h, float umbral) { hip = h; umbral_hip = umbral; }
    hipocampo * hipocampo_activo() const { return hip; }
    // Calcula las firmas que falten (recuerdos anteriores al hipocampo). Devuelve cuántas quedan.
    size_t rellenar_firmas(memoria & m, size_t max);
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
    // Decodifica entradas en lotes de hasta SALIDAS_MAX salidas: llama.cpp reserva un búfer de
    // salidas x vocabulario completo (151 936 en Qwen) y no lo suelta; con 64 salidas son ~40 MB.
    struct entrada { llama_token tok; llama_pos pos; llama_seq_id seq; bool salida; };
    bool decodificar_por_tandas(const std::vector<entrada> & es, const std::function<void(size_t, const float *)> & f);
    void limpiar();   // vacía el KV del motor (y la caché de prefijo)

    llama_model *       model = nullptr;
    const llama_vocab * vocab = nullptr;
    llama_context *     ctx   = nullptr;
    int                 n_vocab = 0;
    std::vector<region> regs;
    uint64_t            huella_ = 0;
    uint64_t            huella_vieja_ = 0;   // formato anterior, aceptada al cargar
    nai_cifrado::clave  cla;
    bool                con_modelo = false;
    hipocampo *         hip = nullptr;
    float               umbral_hip = 0.5f;
    std::vector<uint32_t> cum;
    std::vector<llama_token> en_kv;   // tokens que hay ahora en el KV: el prefijo común se reutiliza
    std::vector<double>      calib;   // puntaje de cada región con texto vacío
    std::vector<llama_token> cab_tok; // encabezado fijo del clasificador (instrucción + temas)
    std::vector<uint8_t>     cab_kv;  // su KV: restaurarlo es una copia, no volver a evaluarlo
    std::atomic<size_t>      n_rechazados{0};
    std::recursive_mutex     mtx_ctx;   // el contexto del modelo: lo usa el fondo; buscar recuerdos no lo necesita
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
        llama_model * modelo_hipocampo = nullptr;   // embeddings (opcional): relacionar por significado
        float       umbral_hipocampo = 0.5f;
        size_t      parte_chars      = 8000;        // un documento más largo se parte en tareas de este tamaño
        size_t      cola_max_chars   = 8u << 20;    // texto pendiente máximo (más: 429, se reintenta después)
    };

    servicio(llama_model * model, const config & cfg);
    ~servicio();
    bool ok() const { return mot && mot->ok(); }
    int  turnos() const { return cfg.turnos; }
    bool es_pregunta_de_codigo(const std::string & texto);   // para enviar al especialista

    // Antes de responder (sincrónico): recuerdos relevantes + turnos anteriores.
    // es_dueno = false (invitado): no recibe recuerdos privados.
    contexto_recordado preparar(uint64_t sesion, const std::string & consulta, bool es_dueno = true);

    // Después de responder (asíncrono): guarda el turno en la memoria de la sesión.
    // ensenar = true (solo el dueño): lo que dijo el usuario también entra a la memoria de la entidad
    // (la biblioteca), si es conocimiento o sabiduría; privado = forzar que quede privado.
    void registrar_turno(uint64_t sesion, const std::string & usuario, const std::string & asistente,
                         bool ensenar = false, bool privado = false);

    // Aprender un texto en la biblioteca. No se procesa de golpe: se parte en tareas de parte_chars
    // que se guardan en disco (DIR/cola) y se procesan trozo a trozo, cediendo el paso a la conversación.
    // Sobrevive a reinicios. Devuelve las partes encoladas, o -1 si la cola está llena.
    int aprender(const std::string & texto, const std::string & fuente, const std::string & region = "", bool privado = false);
    int indice_region(const std::string & nombre) const;  // -1 si no existe

    std::string estado_json();  // estado de la biblioteca y del trabajo pendiente (JSON); no espera al motor
    std::string tareas_json();  // partes pendientes de aprender (JSON)

    // Regla anti-congelamiento: el servidor avisa cuándo empieza y termina cada petición. El trabajo
    // de fondo (codificar turnos, aprender documentos) espera mientras haya alguna en curso, para no
    // competir por CPU/GPU con la respuesta (con un tope de espera para que la cola no se atore).
    void peticion(bool empieza) { activas.fetch_add(empieza ? 1 : -1); }

private:
    memoria & sesion_mem(uint64_t sesion);   // requiere mtx
    void apartar_ilegible(const std::string & ruta) const;   // un .mem que no se pudo leer no se sobrescribe
    void actualizar_estado();                // requiere mtx: copia de las estadísticas para estado_json
    bool trabajar_tarea();                   // un trozo de la cola de aprendizaje; false si no hay
    void guardar_tarea_restante();           // al apagar: lo que falta de la parte en curso vuelve a disco
    std::string ruta_biblioteca() const;
    std::string ruta_sesion(uint64_t sesion) const;
    void purgar();                           // requiere mtx
    void rellenar(uint64_t sesion, bool es_biblioteca);   // firmas faltantes, en tandas (en la cola)
    void encolar(std::function<void()> f);
    void bucle();

    config                           cfg;
    std::unique_ptr<hipocampo>       hip;     // antes que mot: se destruye después
    std::unique_ptr<motor>           mot;
    std::mutex                       mtx;     // protege motor y memorias
    std::map<uint64_t, memoria>      sesiones;
    memoria                          biblioteca;
    int64_t                          t_purga = 0;

    std::mutex                         mtx_cola;
    std::condition_variable            cv;
    std::deque<std::function<void()>>  cola;        // turnos y firmas (en RAM, rápidos)
    std::atomic<bool>                  salir{false};
    std::atomic<bool>                  salir_rapido{false};
    std::thread                        hilo;
    std::atomic<int>                   activas{0};  // peticiones generando
    std::atomic<int>                   esperando{0}; // peticiones esperando el candado para recordar

    // cola de aprendizaje en disco: DIR/cola/*.tarea (una parte por archivo)
    struct tarea {
        std::string ruta, fuente;
        int         region = -1;
        bool        privado = false;
        std::vector<std::string> trozos;
        size_t      hecho = 0;
        bool        cargada = false;
        size_t      chars = 0;
    };
    std::deque<tarea>                  tareas;      // protegida por mtx_cola
    std::atomic<size_t>                cola_chars{0};
    std::atomic<size_t>                aprendidos{0};
    uint64_t                           sec_tarea = 0;

    std::mutex                         mtx_estado;   // copia de estadísticas: estado_json no espera al motor
    estadisticas                       est_bib;
    size_t                             est_rech = 0, est_ses = 0;
};

} // namespace nai_mem
