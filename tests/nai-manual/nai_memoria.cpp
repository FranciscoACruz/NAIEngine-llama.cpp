// [NAI] Memoria reconstructiva por regiones (v3): prueba y medición del módulo common/nai-memoria.
//
// Mide, con un modelo real:
//   1. clasificación por el propio modelo (precisión contra la región esperada)
//   2. reconstrucción EXACTA de cada recuerdo y bytes que ocupa (vs. texto y vs. KV cache)
//   3. olvido con memoria fija: lo menos importante baja a IDEA (frase del modelo) y a PISTA
//   4. "volver a llegar": consultas que encuentran sus recuerdos por las claves (GPS)
//   5. cruce: un concepto activa recuerdos de regiones distintas y el modelo propone una idea nueva
//   6. aprender de un archivo o URL (opcional)
//
// Uso:
//   nai_memoria -m modelo.gguf [-t 4] [--memoria BYTES] [-c "lápiz"] [-f recuerdos.txt] [--compacta] [--diag]
//               [--aprender archivo|URL] [--consulta "pregunta"] [--guardar mem.bin] [--cargar mem.bin]
//               [-r regiones.txt] [--escribir-regiones regiones.txt]
//   nai_memoria -m modelo.gguf --cargar DIR/biblioteca.mem --exportar recuerdos.jsonl [-r regiones.txt]
//               (reconstruye cada recuerdo y lo escribe en JSONL para el reentrenamiento)
//   --clave ARCHIVO   clave de --nai-cifrado para leer/escribir memorias cifradas

#include "llama.h"
#include "nai-lector.h"
#include "nai-memoria.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace nai_mem;

struct ejemplo { const char * texto; const char * region; };

static std::vector<ejemplo> recuerdos_por_defecto() {
    return {
        { "Darse cuenta de que uno está pensando es distinto de simplemente pensar; esa mirada interior es la conciencia.", "conciencia" },
        { "Antes de usar los datos de un cliente para entrenar un sistema, hay que preguntarse si él estaría de acuerdo.", "ética" },
        { "Para Rosa, caminar en silencio por el cerro al amanecer es una forma de oración sin palabras.", "espiritualidad" },
        { "Cuando Pedro perdió su trabajo, lo que más le ayudó fue que su hermana lo escuchara sin darle consejos.", "empatía" },
        { "El miedo que no se nombra suele aparecer en los sueños como persecuciones o caídas.", "psique" },
        { "Si cambias cada tabla de un barco una por una, ¿sigue siendo el mismo barco al final?", "filosofía" },
        { "Si todos los mensajes firmados son auténticos y este mensaje no es auténtico, entonces no está firmado.", "lógica" },
        { "El profesor Ruiz resolvió la demostración a lápiz, dividiendo el problema en dos casos.", "matemáticas" },
        { "Ruiz dice que el papel es una herramienta para plasmar lo que piensa antes de saber si es verdad.", "filosofía" },
        { "Un objeto cae más lento en el aire que en el vacío porque la resistencia del aire se opone a su movimiento.", "física" },
        { "Al mezclar bicarbonato con vinagre se libera dióxido de carbono, por eso aparecen burbujas.", "química" },
        { "Las raíces de una planta crecen hacia donde hay más humedad, aunque tengan que rodear una piedra.", "biología" },
        { "El puente se diseñó para soportar el triple de la carga máxima esperada, por seguridad.", "ingeniería" },
        { "Una palanca larga permite mover una roca pesada con poca fuerza, a cambio de recorrer más distancia.", "mecánica" },
        { "El ala de un avión es más curva por arriba, y esa forma ayuda a generar sustentación.", "aerodinámica" },
        { "Una contraseña hecha de cuatro palabras al azar es más difícil de adivinar que una corta con símbolos.", "ciberseguridad" },
        { "Si te pierdes en el monte, primero busca refugio y agua; la comida puede esperar varios días.", "supervivencia" },
        { "Rotar el maíz con frijol cada temporada devuelve nitrógeno al suelo.", "agricultura" },
        { "Para Lucía el lápiz es donde empieza todo: traza la sombra del rostro antes de pintar el retrato.", "arte" },
        { "A veces, en el margen del cuaderno, Lucía escribe a lápiz un poema sobre la persona que dibuja.", "arte" },
        { "La abuela preparaba té de manzanilla con miel para calmar el dolor de estómago.", "medicina tradicional" },
        { "Juan abrió una cuenta de ahorro el 3 de marzo con 2,500 pesos y deposita 400 cada quincena.", "finanzas personales" },
        { "Decidió dejar la mitad de sus ahorros a plazo fijo hasta diciembre.", "finanzas personales" },
        { "La receta de la abuela no tiene medidas exactas: la harina se mide a ojo.", "cocina" },
        { "El sábado compraron un refrigerador nuevo porque el anterior hacía mucho ruido.", "compras" },
        { "El equipo local perdió el domingo por un gol en el último minuto.", "deportes" },
        { "Para renovar el pasaporte hay que sacar cita en línea y llevar una identificación.", "trámites" },
        { "Anoche vieron una serie de detectives hasta muy tarde.", "entretenimiento" },
    };
}

static void escribir_regiones(const std::string & ruta, const std::vector<region> & rs) {
    std::ofstream f(ruta);
    f << "# peso|tema|semilla   (peso 1 = general, 3 = prioritario; más alto = se conserva más)\n";
    for (auto & r : rs) f << r.peso << "|" << r.nombre << "|" << r.semilla << "\n";
}

static std::string json_esc(const std::string & s) {
    std::string r;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  r += "\\\""; break;
            case '\\': r += "\\\\"; break;
            case '\n': r += "\\n"; break;
            case '\r': r += "\\r"; break;
            case '\t': r += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); r += b; }
                else r += (char) c;
        }
    }
    return r;
}

static int indice_region(const std::vector<region> & rs, const std::string & n) {
    for (size_t i = 0; i < rs.size(); ++i) if (rs[i].nombre == n) return (int) i;
    return -1;
}

int main(int argc, char ** argv) {
    std::string ruta, ruta_regiones, ruta_recuerdos, escribir, concepto = "lápiz", aprender, guardar, cargar, exportar, ruta_clave;
    std::vector<std::string> consultas = { "¿Cuánto deposita Juan en su cuenta de ahorro?", "¿Qué le ayudó a Pedro cuando perdió su trabajo?",
                                           "¿Cómo se genera la sustentación en un avión?", "¿Qué remedio usaba la abuela para el estómago?" };
    bool consultas_propias = false;
    long memoria_bytes = 0;
    int n_threads = 4;
    bool compacta = false; // [NAI] --compacta: codificar con el modelo
    bool diag     = false; // [NAI] --diag: por qué el clasificador elige cada región
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto sig = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "-m") ruta = sig();
        else if (a == "-t") n_threads = atoi(sig().c_str());
        else if (a == "-c") concepto = sig();
        else if (a == "-r") ruta_regiones = sig();
        else if (a == "-f") ruta_recuerdos = sig();
        else if (a == "--memoria") memoria_bytes = atol(sig().c_str());
        else if (a == "--escribir-regiones") escribir = sig();
        else if (a == "--aprender") aprender = sig();
        else if (a == "--guardar") guardar = sig();
        else if (a == "--cargar") cargar = sig();
        else if (a == "--exportar") exportar = sig();
        else if (a == "--clave") ruta_clave = sig();
        else if (a == "--compacta") compacta = true;
        else if (a == "--diag") diag = true;
        else if (a == "--consulta") { if (!consultas_propias) consultas.clear(); consultas_propias = true; consultas.push_back(sig()); }
        else { fprintf(stderr, "opción desconocida: %s\n", a.c_str()); return 1; }
    }
    std::vector<region> regs = ruta_regiones.empty() ? regiones_por_defecto() : leer_regiones(ruta_regiones);
    if (!escribir.empty()) { escribir_regiones(escribir, regs); printf("regiones escritas en %s\n", escribir.c_str()); return 0; }
    if (ruta.empty()) { fprintf(stderr, "uso: %s -m modelo.gguf [opciones]\n", argv[0]); return 1; }

    llama_backend_init();
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(ruta.c_str(), mp);
    if (!model) { fprintf(stderr, "no se pudo cargar %s\n", ruta.c_str()); return 1; }

    motor mot(model, n_threads, regs);
    mot.usar_codificacion_modelo(compacta);
    printf("codificación: %s\n", compacta ? "con el modelo (compacta)" : "diccionario de tokens (portable)");
    if (!mot.ok()) { fprintf(stderr, "no se pudo crear el contexto\n"); return 1; }
    regs = mot.regiones();
    if (!ruta_clave.empty()) {
        if (!std::ifstream(ruta_clave)) { fprintf(stderr, "no existe la clave %s\n", ruta_clave.c_str()); return 1; }
        nai_cifrado::clave cla;
        std::string aviso;
        if (!nai_cifrado::cargar_o_crear(ruta_clave, cla, aviso)) { fprintf(stderr, "%s\n", aviso.c_str()); return 1; }
        mot.usar_clave(cla);
    }

    memoria mem;
    mem.presupuesto = 1u << 30; // sin límite mientras se mide el costo exacto
    if (!cargar.empty()) {
        if (mot.cargar(mem, cargar)) printf("memoria cargada de %s: %zu recuerdos, %zu bytes\n", cargar.c_str(), mem.recuerdos.size(), mem.bytes());
        else printf("no se pudo cargar %s (otro modelo, otras regiones o archivo inválido)\n", cargar.c_str());
    }

    // ---------------------------------------------------------------- exportar (para reentrenar)
    if (!exportar.empty()) {
        if (cargar.empty() || mem.recuerdos.empty()) {
            fprintf(stderr, "--exportar requiere --cargar con una memoria que tenga recuerdos\n");
            return 1;
        }
        std::ofstream f(exportar);
        size_t n = 0, danados = 0;
        for (auto & r : mem.recuerdos) {
            const std::string t = mot.texto(r);
            if (r.danado || t.empty()) { danados++; continue; }
            f << "{\"region\":\"" << json_esc(regs[r.region].nombre) << "\",\"peso\":" << regs[r.region].peso
              << ",\"nivel\":\"" << nombre_nivel(r.nivel) << "\",\"rol\":\"" << nombre_rol(r.rol)
              << "\",\"usos\":" << (int) r.usos << ",\"texto\":\"" << json_esc(t) << "\"}\n";
            n++;
        }
        printf("exportados %zu recuerdos a %s (%zu no se pudieron reconstruir)\n", n, exportar.c_str(), danados);
        llama_model_free(model);
        llama_backend_free();
        return danados == mem.recuerdos.size() ? 1 : 0;
    }

    // ---------------------------------------------------------------- 1. clasificación y 2. exactitud
    std::vector<ejemplo> ejemplos;
    std::vector<std::string> propios;
    if (!ruta_recuerdos.empty()) {
        std::ifstream f(ruta_recuerdos);
        std::string l;
        while (std::getline(f, l)) if (l.size() > 8) propios.push_back(l);
        for (auto & p : propios) ejemplos.push_back({ p.c_str(), "" });
    } else if (cargar.empty()) {
        ejemplos = recuerdos_por_defecto();
    }

    if (diag) {
        printf("\n== Diagnóstico del clasificador ==\n");
        const std::vector<double> cal = mot.puntuar_regiones("N/A");
        int k = 0;
        for (auto & e : ejemplos) {
            if (!e.region[0] || k++ >= 4) continue;
            const std::string libre = mot.generar(mot.instruccion_clasificar(e.texto), 12);
            const std::vector<double> sc = mot.puntuar_regiones(e.texto);
            std::vector<size_t> o1(sc.size()), o2(sc.size());
            for (size_t i = 0; i < sc.size(); ++i) o1[i] = o2[i] = i;
            std::sort(o1.begin(), o1.end(), [&](size_t a, size_t b) { return sc[a] > sc[b]; });
            std::sort(o2.begin(), o2.end(), [&](size_t a, size_t b) { return sc[a] - cal[a] > sc[b] - cal[b]; });
            printf("  [%s] %s\n    respuesta libre: \"%s\"\n    sin calibrar:", e.region, e.texto, libre.c_str());
            for (int j = 0; j < 3; ++j) printf(" %s %.2f", regs[o1[j]].nombre.c_str(), sc[o1[j]]);
            printf("\n    calibrada:   ");
            for (int j = 0; j < 3; ++j) printf(" %s %.2f", regs[o2[j]].nombre.c_str(), sc[o2[j]] - cal[o2[j]]);
            const int esp = indice_region(regs, e.region);
            if (esp >= 0) printf("\n    esperada: %s  bruto %.2f  calibrado %.2f", e.region, sc[esp], sc[esp] - cal[esp]);
            printf("\n");
        }
    }
    printf("\n== 1-2. Clasificación por el modelo y reconstrucción exacta ==\n");
    int aciertos = 0, con_etiqueta = 0, exactos = 0;
    size_t bytes_texto = 0;
    const int64_t t0 = llama_time_us();
    for (auto & e : ejemplos) {
        const int reg = mot.clasificar(e.texto);
        const size_t antes = mem.recuerdos.size();
        mot.agregar(mem, e.texto, reg, DOCUMENTO);
        if (mem.recuerdos.size() == antes) { printf("  (no se pudo codificar) %s\n", e.texto); continue; }
        recuerdo & r = mem.recuerdos.back();
        r.cache.clear(); // forzar la reconstrucción desde los bytes
        const std::string rec = mot.texto(r);
        const bool exacto = !r.danado && rec == std::string(e.texto);
        exactos += exacto;
        bytes_texto += strlen(e.texto);
        const int esperado = e.region[0] ? indice_region(regs, e.region) : -1;
        if (esperado >= 0) { con_etiqueta++; aciertos += (esperado == reg); }
        printf("  %-20s %s %3zu B  %s%s\n", regs[reg].nombre.c_str(), exacto ? "exacto" : "DIFIERE", r.bytes(),
               esperado >= 0 && esperado != reg ? ("[esperado: " + std::string(e.region) + "] ").c_str() : "", e.texto);
    }
    const double seg = (llama_time_us() - t0) / 1e6;
    if (!ejemplos.empty()) {
        printf("\n  reconstrucción exacta: %d/%zu\n", exactos, ejemplos.size());
        if (con_etiqueta) printf("  clasificación correcta: %d/%d (%.0f %%)\n", aciertos, con_etiqueta, 100.0 * aciertos / con_etiqueta);
        if (con_etiqueta) { // A/B: el mismo clasificador sin calibración, y lo que cuesta cada uno
            int sin = 0, n = 0;
            const int64_t a0 = llama_time_us();
            for (auto & e : ejemplos) {
                const int esp = e.region[0] ? indice_region(regs, e.region) : -1;
                if (esp >= 0) { ++n; sin += mot.clasificar(e.texto, false) == esp; }
            }
            const double ms_sin = (llama_time_us() - a0) / 1e3 / std::max(1, n);
            const int64_t a1 = llama_time_us();
            for (auto & e : ejemplos) if (e.region[0]) mot.clasificar(e.texto, true);
            const double ms_con = (llama_time_us() - a1) / 1e3 / std::max(1, n);
            printf("  clasificación sin calibrar:    %d/%d (%.0f %%), %.0f ms cada una\n", sin, n, 100.0 * sin / n, ms_sin);
            printf("  clasificación calibrada:       %d/%d (%.0f %%), %.0f ms cada una (la que se usa)\n", aciertos, con_etiqueta,
                   100.0 * aciertos / con_etiqueta, ms_con);
        }
        const size_t b = mem.bytes();
        const double kv = 0; (void) kv;
        printf("  memoria: %zu B para %zu B de texto (%.1fx menos); %.2f s por recuerdo\n", b, bytes_texto,
               b ? (double) bytes_texto / b : 0.0, seg / ejemplos.size());
    }

    // ---------------------------------------------------------------- 6. aprender de archivo o URL
    if (!aprender.empty()) {
        printf("\n== Aprender de %s ==\n", aprender.c_str());
        const bool es_url = aprender.rfind("http://", 0) == 0 || aprender.rfind("https://", 0) == 0;
        const nai_lector::resultado lr = es_url ? nai_lector::leer_url(aprender) : nai_lector::leer_archivo(aprender);
        if (!lr.ok) {
            printf("  error: %s\n", lr.error.c_str());
        } else {
            const int64_t ta = llama_time_us();
            const int n = mot.agregar(mem, lr.texto, -1, DOCUMENTO);
            printf("  tipo %s: %zu caracteres -> %d recuerdos en %.1f s\n", lr.tipo.c_str(), lr.texto.size(), n, (llama_time_us() - ta) / 1e6);
            std::map<std::string, int> por_region;
            for (size_t i = mem.recuerdos.size() - n; i < mem.recuerdos.size(); ++i) por_region[regs[mem.recuerdos[i].region].nombre]++;
            for (auto & kv : por_region) printf("    %-20s %d\n", kv.first.c_str(), kv.second);
        }
    }

    // ---------------------------------------------------------------- 3. olvido con memoria fija
    const size_t total = mem.bytes();
    mem.presupuesto = memoria_bytes > 0 ? (size_t) memoria_bytes : total / 2;
    printf("\n== 3. Memoria fija de %zu B (ocupado: %zu B) ==\n", mem.presupuesto, total);
    const int64_t t1 = llama_time_us();
    mot.ajustar(mem);
    estadisticas e = mot.medir(mem);
    printf("  exacto %zu, idea %zu, pista %zu (dañados %zu) -> %zu B en %.1f s\n", e.n_exacto, e.n_idea, e.n_pista,
           e.n_danados, e.bytes, (llama_time_us() - t1) / 1e6);
    int mostrados = 0;
    for (auto & r : mem.recuerdos) {
        if (r.nivel == EXACTO || mostrados >= 8) continue;
        printf("  %-6s %-20s %3zu B  %s\n", nombre_nivel(r.nivel), regs[r.region].nombre.c_str(), r.bytes(), mot.texto(r).c_str());
        mostrados++;
    }

    // ---------------------------------------------------------------- 4. volver a llegar
    printf("\n== 4. Volver a llegar (consultas) ==\n");
    for (auto & q : consultas) {
        int n = 0;
        const std::string b = mot.recordar({ &mem }, q, 3, 1200, &n);
        printf("  ? %s\n", q.c_str());
        if (n == 0) { printf("    (sin recuerdos)\n"); continue; }
        size_t p = b.find('\n');
        while (p != std::string::npos && p + 1 < b.size()) {
            const size_t f = b.find('\n', p + 1);
            printf("    %s\n", b.substr(p + 1, f - p - 1).c_str());
            p = f;
        }
    }

    // ---------------------------------------------------------------- 5. cruce de regiones
    printf("\n== 5. Cruce: \"%s\" ==\n", concepto.c_str());
    int n = 0;
    const std::string bloque = mot.recordar({ &mem }, concepto, 3, 1500, &n);
    if (n >= 2) {
        printf("%s", bloque.substr(bloque.find('\n') + 1).c_str());
        const std::string idea = mot.generar(
            "Estos recuerdos vienen de temas distintos:\n" + bloque.substr(bloque.find('\n') + 1) +
            "Combínalos en una sola idea nueva y original, en una o dos frases.", 96);
        printf("  idea nueva: %s\n", idea.c_str());
    } else {
        printf("  (el concepto activó %d recuerdos; se necesitan al menos 2 de regiones distintas)\n", n);
    }

    if (!guardar.empty()) {
        if (mot.guardar(mem, guardar)) printf("\nmemoria guardada en %s (%zu B de recuerdos)\n", guardar.c_str(), mem.bytes());
        else printf("\nno se pudo guardar %s\n", guardar.c_str());
    }

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
