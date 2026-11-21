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
    std::printf("[test-nai] triaje OK\n");
}

static void test_sesiones() {
    // hash: estable, distinto por id y nunca valores reservados
    NAI_CHECK(nai::hash_sesion("usuario-1") == nai::hash_sesion("usuario-1"));
    NAI_CHECK(nai::hash_sesion("usuario-1") != nai::hash_sesion("usuario-2"));
    NAI_CHECK(nai::hash_sesion("") != NAI_EMPTY_HANDLE);

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

int main() {
    test_entropia();
    test_mapa_masa();
    test_telemetria();
    test_triaje();
    test_sesiones();
    std::printf("[test-nai] PASS\n");
    return 0;
}
