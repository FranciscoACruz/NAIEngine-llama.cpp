// [NAI] Pruebas unitarias del motor NAI que no requieren un modelo:
//   - entropía normalizada
//   - mapa de masa semántica (registro, búsqueda de ventana, recorrido)
//   - ring y hub de telemetría
//   - normalización y modos del TriageEngine

#include "../src/inercia_latente.hpp"
#include "../src/nai-telemetry-ring.hpp"
#include "../src/nai-telemetry-hub.hpp"
#include "../common/TriageWrapper.h"
#include "../src/nai-session-table.hpp"
#include "../common/nai-memoria.h"
#include "../common/nai-lector.h"
#include "../common/nai-cifrado.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#define NAI_CHECK(cond)                                                          \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "%s:%d: fallo: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

static bool cerca(float a, float b, float tol = 1e-4f) { return std::fabs(a - b) <= tol; }

static void test_entropia() {
    std::vector<float> uniforme(1000, 0.5f);
    NAI_CHECK(cerca(nai::entropia_normalizada(uniforme.data(), uniforme.size()), 1.0f));

    std::vector<float> pico(1000, 0.0f);
    pico[3] = 100.0f;
    NAI_CHECK(nai::entropia_normalizada(pico.data(), pico.size()) < 1e-3f);

    std::vector<float> invalidos(4, NAN);
    NAI_CHECK(nai::entropia_normalizada(invalidos.data(), invalidos.size()) == nai::MASA_PROTEGIDA);
    NAI_CHECK(nai::entropia_normalizada(nullptr, 10) == nai::MASA_PROTEGIDA);
    std::printf("[test-nai] entropia OK\n");
}

static void test_sorpresa() {
    // distribución uniforme: cualquier token tiene sorpresa ln(n) -> 1.0
    std::vector<float> uniforme(1000, 0.5f);
    NAI_CHECK(cerca(nai::sorpresa_normalizada(uniforme.data(), uniforme.size(), 7), 1.0f));

    // pico en el token 3: el token 3 es casi seguro (masa ~0), cualquier otro es muy sorprendente (1.0)
    std::vector<float> pico(1000, 0.0f);
    pico[3] = 100.0f;
    NAI_CHECK(nai::sorpresa_normalizada(pico.data(), pico.size(), 3) < 1e-3f);
    NAI_CHECK(cerca(nai::sorpresa_normalizada(pico.data(), pico.size(), 4), 1.0f));

    // valor intermedio exacto: p = 0.5 entre dos tokens de un vocabulario de 4 -> ln2/ln4 = 0.5
    std::vector<float> dos = {10.0f, 10.0f, -1e9f, -1e9f};
    NAI_CHECK(cerca(nai::sorpresa_normalizada(dos.data(), dos.size(), 0), 0.5f));

    // datos inválidos
    NAI_CHECK(nai::sorpresa_normalizada(nullptr, 10, 0) == nai::MASA_PROTEGIDA);
    NAI_CHECK(nai::sorpresa_normalizada(uniforme.data(), uniforme.size(), -1) == nai::MASA_PROTEGIDA);
    NAI_CHECK(nai::sorpresa_normalizada(uniforme.data(), uniforme.size(), 1000) == nai::MASA_PROTEGIDA);
    std::vector<float> nan4(4, NAN);
    NAI_CHECK(nai::sorpresa_normalizada(nan4.data(), nan4.size(), 0) == nai::MASA_PROTEGIDA);
    // corte del lse: error despreciable frente al cálculo exacto en doble precisión
    {
        std::vector<float> l(152000);
        uint32_t semilla = 12345;
        for (auto & v : l) { semilla = semilla * 1664525u + 1013904223u; v = (float) ((semilla >> 8) % 20000) / 1000.0f - 10.0f; }
        l[42] = 25.0f; // un token dominante
        const double exacto = nai::log_sum_exp(l.data(), l.size());
        const float  rapido = nai::log_sum_exp_rapido(l.data(), l.size());
        NAI_CHECK(std::fabs(exacto - rapido) < 0.02);
        for (int32_t t : {0, 42, 1000, 151999}) {
            const float m_exacta = (float) std::min(1.0, std::max(0.0, (exacto - l[t]) / std::log(152000.0)));
            NAI_CHECK(std::fabs(nai::sorpresa_normalizada(l.data(), l.size(), t) - m_exacta) < 0.002f);
        }
    }
    // exp_rapido: error relativo < 2e-6 en [-16, 0]
    for (float x = -16.0f; x <= 0.0f; x += 0.01f) {
        NAI_CHECK(std::fabs(nai::exp_rapido(x) / std::exp((double) x) - 1.0) < 2e-6);
    }
    // NaN sueltos se ignoran; todo NaN o todo -inf => masa protegida
    {
        std::vector<float> con_nan = {1.0f, NAN, 3.0f, 2.0f};
        NAI_CHECK(std::isfinite(nai::log_sum_exp_rapido(con_nan.data(), con_nan.size())));
        std::vector<float> menos_inf(8, -INFINITY);
        NAI_CHECK(nai::sorpresa_normalizada(menos_inf.data(), menos_inf.size(), 0) == nai::MASA_PROTEGIDA);
    }
    std::printf("[test-nai] sorpresa OK\n");
}

static void test_pendientes() {
    nai::MapaMasaSemantica m;
    m.init(2, 64);
    std::vector<float> pico(16, 0.0f);
    pico[5] = 100.0f;

    // la fila calculada en pos 9 predice pos 10: la masa va a pos 10, no a 9
    m.guardar_pendiente(0, 10, pico.data(), pico.size());
    NAI_CHECK(m.pos_pendiente(0) == 10);
    NAI_CHECK(!m.resolver_pendiente(0, 11, 5)); // otra posición: no resuelve
    NAI_CHECK(!m.resolver_pendiente(1, 10, 5)); // otra secuencia: no resuelve
    NAI_CHECK(m.resolver_pendiente(0, 10, 5));  // llegó el token predicho
    NAI_CHECK(m.obtener(0, 10) < 1e-3f);
    NAI_CHECK(m.obtener(0, 9) == nai::MASA_PROTEGIDA);
    NAI_CHECK(m.pos_pendiente(0) == -1);        // se consume una sola vez

    // un token inesperado conserva masa alta
    m.guardar_pendiente(0, 11, pico.data(), pico.size());
    NAI_CHECK(m.resolver_pendiente(0, 11, 2));
    NAI_CHECK(cerca(m.obtener(0, 11), 1.0f));

    // el desalojo recorre la predicción pendiente igual que el KV
    m.guardar_pendiente(0, 30, pico.data(), pico.size());
    m.quitar_y_recorrer(0, 20, 4, 31);
    NAI_CHECK(m.pos_pendiente(0) == 26);
    m.guardar_pendiente(0, 22, pico.data(), pico.size());
    m.quitar_y_recorrer(0, 20, 4, 31);          // la posición pendiente fue desalojada
    NAI_CHECK(m.pos_pendiente(0) == -1);

    // limpiar descarta pendientes
    m.guardar_pendiente(1, 5, pico.data(), pico.size());
    m.limpiar(1);
    NAI_CHECK(m.pos_pendiente(1) == -1);

    // posición fuera de rango: no se guarda
    m.guardar_pendiente(0, 999, pico.data(), pico.size());
    NAI_CHECK(m.pos_pendiente(0) == -1);
    std::printf("[test-nai] predicciones pendientes OK\n");
}

static void test_mapa_masa() {
    nai::MapaMasaSemantica m;
    NAI_CHECK(!m.activo());
    NAI_CHECK(m.buscar_ventana_minima(0, 0, 10, 2) == -1);

    m.init(2, 64);
    NAI_CHECK(m.activo());
    NAI_CHECK(m.obtener(0, 5) == nai::MASA_PROTEGIDA);
    NAI_CHECK(m.obtener(5, 5) == nai::MASA_PROTEGIDA);  // secuencia fuera de rango
    NAI_CHECK(m.obtener(0, 999) == nai::MASA_PROTEGIDA); // posición fuera de rango

    // secuencias independientes
    m.registrar(0, 10, 0.1f);
    NAI_CHECK(cerca(m.obtener(0, 10), 0.1f));
    NAI_CHECK(m.obtener(1, 10) == nai::MASA_PROTEGIDA);

    // la ventana de menor masa es [20, 24)
    for (int p = 20; p < 24; ++p) m.registrar(0, p, 0.0f);
    NAI_CHECK(m.buscar_ventana_minima(0, 0, 40, 4) == 20);
    NAI_CHECK(m.buscar_ventana_minima(0, 22, 40, 4) != 20); // respeta p_min
    NAI_CHECK(m.buscar_ventana_minima(0, 0, 3, 4) == -1);   // no cabe

    // quitar [20, 24) y recorrer [24, 40) -> lo que estaba en 30 pasa a 26
    m.registrar(0, 30, 0.7f);
    m.quitar_y_recorrer(0, 20, 4, 40);
    NAI_CHECK(cerca(m.obtener(0, 26), 0.7f));
    NAI_CHECK(m.obtener(0, 36) == nai::MASA_PROTEGIDA); // cola liberada
    NAI_CHECK(cerca(m.obtener(0, 10), 0.1f));            // antes de p0 no cambia

    m.limpiar(0);
    NAI_CHECK(m.obtener(0, 10) == nai::MASA_PROTEGIDA);
    std::printf("[test-nai] mapa de masa OK\n");
}

static void test_telemetria() {
    struct Ev { uint64_t a; uint32_t b; uint32_t c; };
    nai::NaiTelemetryRing<Ev> ring(8);
    for (uint32_t i = 0; i < 7; ++i) NAI_CHECK(ring.try_push({i, i, 0}));
    NAI_CHECK(!ring.try_push({99, 99, 0}));
    Ev out;
    for (uint32_t i = 0; i < 7; ++i) { NAI_CHECK(ring.try_pop(out)); NAI_CHECK(out.b == i); }
    NAI_CHECK(!ring.try_pop(out));

    nai::NaiTelemetryHub hub(1024);
    nai::NaiTelemetryEvent ev{};
    NAI_CHECK(!hub.emit(ev)); // sin consumidor: se descarta

    std::atomic<int> n{0};
    hub.set_callback([&n](const nai::NaiTelemetryEvent &) { n.fetch_add(1); });

    // 4 productores concurrentes
    std::vector<std::thread> prod;
    for (int t = 0; t < 4; ++t) {
        prod.emplace_back([&hub]() {
            nai::NaiTelemetryEvent e{};
            for (int i = 0; i < 100; ++i) {
                while (!hub.emit(e)) std::this_thread::yield();
            }
        });
    }
    for (auto & t : prod) t.join();

    for (int i = 0; i < 200 && n.load() < 400; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    hub.stop();
    NAI_CHECK(n.load() == 400);
    std::printf("[test-nai] telemetria OK\n");
}

static void test_triaje() {
    TriageEngine t;
    NAI_CHECK(t.procesar_prompt("ÁYUDAME con algo").modo == ModoTriaje::INDAGACION_CORTA);
    NAI_CHECK(t.procesar_prompt("ayudame con la FUNCIÓN").modo == ModoTriaje::DIRECTO_AUTOPISTA); // acento normalizado
    NAI_CHECK(t.procesar_prompt("explica la fotosintesis").modo == ModoTriaje::DIRECTO_AUTOPISTA);
    NAI_CHECK(t.procesar_prompt("crea algo bonito para la fiesta de mi hermana").modo == ModoTriaje::PROPUESTA_HIPOTESIS);
    // el modo hipótesis ya no inyecta supuestos (antes forzaba "C++") en el mensaje
    NAI_CHECK(t.procesar_prompt("crea algo bonito para la fiesta de mi hermana").contexto_inyectado.empty());
    NAI_CHECK(t.procesar_prompt("help me with docker").modo == ModoTriaje::DIRECTO_AUTOPISTA);   // ancla en inglés
    NAI_CHECK(t.procesar_prompt("hazme un script en rust").modo == ModoTriaje::DIRECTO_AUTOPISTA);
    std::printf("[test-nai] triaje OK\n");
}

static void test_sesiones() {
    // hash: estable, distinto por id y nunca valores reservados
    NAI_CHECK(nai::hash_sesion("usuario-1") == nai::hash_sesion("usuario-1"));
    NAI_CHECK(nai::hash_sesion("usuario-1") != nai::hash_sesion("usuario-2"));
    NAI_CHECK(nai::hash_sesion("") != NAI_EMPTY_HANDLE);

    // SipHash-2-4: vector de prueba oficial (clave 00..0f, mensaje vacío) = 0x726fdb47dd0e0e31
    {
        nai::ClaveSesion k{0x0706050403020100ULL, 0x0f0e0d0c0b0a0908ULL};
        NAI_CHECK(nai::siphash24("", 0, k) == 0x726fdb47dd0e0e31ULL);
        // mensaje 00..0e (15 bytes) = 0xa129ca6149be45e5
        unsigned char msg[15];
        for (int i = 0; i < 15; ++i) msg[i] = (unsigned char) i;
        NAI_CHECK(nai::siphash24(msg, sizeof(msg), k) == 0xa129ca6149be45e5ULL);
        // con otra clave el mismo id produce otro handle
        nai::ClaveSesion k2{1, 2};
        NAI_CHECK(nai::hash_sesion("usuario-1", k) != nai::hash_sesion("usuario-1", k2));
    }

    nai::NaiSessionTable t(16);
    auto nueva = [](const std::string & id, int64_t t_act) {
        NaiSession s;
        s.session_handle = nai::hash_sesion(id);
        s.last_activity  = t_act;
        return s;
    };

    // insertar / buscar / actualizar
    t.guardar(nueva("a", 100));
    t.guardar(nueva("b", 200));
    NaiSession out;
    NAI_CHECK(t.buscar(nai::hash_sesion("a"), out) && out.last_activity == 100);
    t.guardar(nueva("a", 150));
    NAI_CHECK(t.buscar(nai::hash_sesion("a"), out) && out.last_activity == 150);
    NAI_CHECK(t.size() == 2);
    NAI_CHECK(!t.buscar(nai::hash_sesion("zzz"), out));

    // eliminar deja lápida y no rompe la búsqueda de las demás
    NAI_CHECK(t.eliminar(nai::hash_sesion("a")));
    NAI_CHECK(!t.buscar(nai::hash_sesion("a"), out));
    NAI_CHECK(t.buscar(nai::hash_sesion("b"), out));
    NAI_CHECK(t.size() == 1);

    // expiración por TTL: con ttl=100 y ahora=301 expira "b" (200 + 100 <= 301)
    t.guardar(nueva("c", 290));
    std::vector<uint64_t> expiradas;
    const size_t n = t.expirar(301, 100, [&](const NaiSession & s) { expiradas.push_back(s.session_handle); });
    NAI_CHECK(n == 1 && expiradas.size() == 1 && expiradas[0] == nai::hash_sesion("b"));
    NAI_CHECK(t.buscar(nai::hash_sesion("c"), out));

    // crecimiento: 5000 sesiones en una tabla que empezó con 16 posiciones
    for (int i = 0; i < 5000; ++i) t.guardar(nueva("s" + std::to_string(i), i));
    NAI_CHECK(t.size() == 5001);
    for (int i = 0; i < 5000; i += 97) {
        NAI_CHECK(t.buscar(nai::hash_sesion("s" + std::to_string(i)), out) && out.last_activity == i);
    }
    // borrar muchas e insertar de nuevo (reutiliza lápidas / reconstruye)
    for (int i = 0; i < 5000; i += 2) t.eliminar(nai::hash_sesion("s" + std::to_string(i)));
    for (int i = 5000; i < 7000; ++i) t.guardar(nueva("s" + std::to_string(i), i));
    NAI_CHECK(t.size() == 1 + 2500 + 2000);
    NAI_CHECK(t.buscar(nai::hash_sesion("s4999"), out));
    NAI_CHECK(!t.buscar(nai::hash_sesion("s4998"), out));
    std::printf("[test-nai] sesiones OK (capacidad final %zu)\n", t.capacidad());
}

// [NAI] Memoria v3: claves (GPS), troceo de documentos y lectura de HTML. No requiere modelo.
static void test_memoria() {
    using namespace nai_mem;
    // misma raíz y sin acentos: "ahorros" y "ahorró" llevan a "ahorro"
    NAI_CHECK(clave_palabra("ahorros") == clave_palabra("Ahorró"));
    NAI_CHECK(clave_palabra("Cinchona") == clave_palabra("cinchona"));
    // relleno y palabras cortas no son claves
    NAI_CHECK(clave_palabra("porque") == 0 && clave_palabra("con") == 0 && clave_palabra("Cuándo") == 0);
    // los signos ¿ ¡ « » no se pegan a las palabras
    NAI_CHECK(claves_de("¿Qué remedio?") == claves_de("Qué remedio"));
    NAI_CHECK(claves_de("«quinina»").size() == 1 && claves_de("«quinina»")[0] == clave_palabra("quinina"));
    // colisión detectada antes de mezclar los bits del hash
    NAI_CHECK(clave_palabra("perdió") != clave_palabra("quincena"));
    // claves sin repetir
    NAI_CHECK(claves_de("lápiz lápiz LÁPIZ").size() == 1);

    // troceo: párrafos y oraciones, ningún trozo excede el máximo, nada se pierde
    std::string largo;
    for (int i = 0; i < 40; ++i) largo += "La oración número " + std::to_string(i) + " habla del suelo y del agua. ";
    const std::string doc = "Primer párrafo corto sobre la milpa y el frijol.\n\n" + largo + "\n\nÚltimo párrafo sobre las abejas.";
    const auto trozos = trocear(doc, 300, 10);
    NAI_CHECK(trozos.size() >= 4);
    size_t total = 0;
    for (auto & t : trozos) { NAI_CHECK(t.size() <= 300); total += t.size(); }
    NAI_CHECK(trozos.front().find("milpa") != std::string::npos);
    NAI_CHECK(trozos.back().find("abejas") != std::string::npos);
    NAI_CHECK(total + 60 >= doc.size() - 6); // solo se pierden espacios entre trozos
    NAI_CHECK(trocear("corto", 300, 10).empty()); // menos que el mínimo

    // HTML: sin script, estilo ni comentarios; entidades decodificadas; párrafos separados
    const std::string t = nai_lector::html_a_texto(
        "<html><head><title>x</title><style>p{}</style></head><body><script>var a=1;</script>"
        "<p>Ma&iacute;z &amp; frijol</p><!-- oculto --><p>Caf&#233; &#x41;</p></body></html>");
    NAI_CHECK(t.find("Maíz & frijol") != std::string::npos);
    NAI_CHECK(t.find("Café A") != std::string::npos);
    NAI_CHECK(t.find("var a") == std::string::npos && t.find("oculto") == std::string::npos && t.find("p{}") == std::string::npos);

    // páginas web: se prefiere <article>; menús, pies y líneas sueltas se descartan
    std::string art;
    for (int i = 0; i < 12; ++i) art += "<p>La rotación de cultivos mantiene la fertilidad del suelo y reduce las plagas en la parcela.</p>";
    const auto web = nai_lector::leer_bytes("pagina.html", "<html><body><nav><a>Inicio</a> <a>Tienda</a></nav><div>Suscríbete ya</div>"
        "<article>" + art + "</article><footer>Todos los derechos reservados 2026 empresa ejemplo con aviso legal largo</footer></body></html>");
    NAI_CHECK(web.ok && web.texto.find("rotación de cultivos") != std::string::npos);
    NAI_CHECK(web.texto.find("Tienda") == std::string::npos && web.texto.find("Suscríbete") == std::string::npos &&
              web.texto.find("derechos reservados") == std::string::npos);
    NAI_CHECK(nai_lector::limpiar_web("Inicio\nUna línea con suficientes palabras para ser contenido real.\nOK\n") ==
              "Una línea con suficientes palabras para ser contenido real.\n");

    // lector: texto Latin-1 se convierte a UTF-8; binarios se rechazan
    const auto r1 = nai_lector::leer_bytes("notas.txt", std::string("ma\xEDz"));
    NAI_CHECK(r1.ok && r1.texto == "maíz");
    const auto r2 = nai_lector::leer_bytes("dato.bin", std::string("ab\0cd", 5));
    NAI_CHECK(!r2.ok);
    std::printf("[test-nai] memoria v3 OK (%zu trozos)\n", trozos.size());
}

// [NAI] Cifrado: vector oficial RFC 8439 §2.8.2 (ChaCha20-Poly1305), alteraciones y contexto.
static void test_cifrado() {
    auto hex = [](const std::string & h) { std::string r; for (size_t i = 0; i + 1 < h.size(); i += 2) r += (char) std::stoi(h.substr(i, 2), nullptr, 16); return r; };
    uint8_t key[32];
    for (int i = 0; i < 32; ++i) key[i] = (uint8_t) (0x80 + i);
    const std::string nonce = hex("070000004041424344454647"), aad = hex("50515253c0c1c2c3c4c5c6c7");
    std::string m = "Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.";
    const std::string original = m;
    uint8_t tag[16];
    nai_cifrado::sellar(key, (const uint8_t *) nonce.data(), (const uint8_t *) aad.data(), aad.size(), (uint8_t *) m.data(), m.size(), tag);
    NAI_CHECK(m.substr(0, 16) == hex("d31a8d34648e60db7b86afbc53ef7ec2"));
    NAI_CHECK(std::string((const char *) tag, 16) == hex("1ae10b594f09e26a7e902ecbd0600691"));
    std::string c = m;
    NAI_CHECK(nai_cifrado::abrir(key, (const uint8_t *) nonce.data(), (const uint8_t *) aad.data(), aad.size(), (uint8_t *) c.data(), c.size(), tag));
    NAI_CHECK(c == original);
    c = m; c[5] ^= 1; // un bit alterado
    NAI_CHECK(!nai_cifrado::abrir(key, (const uint8_t *) nonce.data(), (const uint8_t *) aad.data(), aad.size(), (uint8_t *) c.data(), c.size(), tag));

    // archivo completo: contexto (nombre) incorrecto, otra clave o alteración -> no descifra
    nai_cifrado::clave k1, k2;
    for (int i = 0; i < 32; ++i) { k1.k[i] = (uint8_t) i; k2.k[i] = (uint8_t) (i + 1); }
    k1.activa = k2.activa = true;
    const std::string plano = "Usuario: ahorro 400 pesos cada quincena";
    const std::string arch = nai_cifrado::cifrar(k1, plano, "nai-0001.mem");
    std::string out;
    NAI_CHECK(nai_cifrado::es_cifrado(arch) && arch.find("ahorro") == std::string::npos);
    NAI_CHECK(nai_cifrado::descifrar(k1, arch, "nai-0001.mem", out) && out == plano);
    NAI_CHECK(!nai_cifrado::descifrar(k1, arch, "nai-0002.mem", out));
    NAI_CHECK(!nai_cifrado::descifrar(k2, arch, "nai-0001.mem", out));
    std::string alterado = arch; alterado[20] ^= 0x40;
    NAI_CHECK(!nai_cifrado::descifrar(k1, alterado, "nai-0001.mem", out));
    NAI_CHECK(nai_cifrado::cifrar(k1, plano, "x") != nai_cifrado::cifrar(k1, plano, "x")); // nonce aleatorio
    std::printf("%s\n", "[test-nai] cifrado OK (RFC 8439)");
}

int main() {
    test_entropia();
    test_memoria();
    test_cifrado();
    test_sorpresa();
    test_pendientes();
    test_mapa_masa();
    test_telemetria();
    test_triaje();
    test_sesiones();
    std::printf("[test-nai] PASS\n");
    return 0;
}
