#pragma once

// [NAI] Lectura de PDF sin dependencias nuevas.
//
// El modelo solo ve texto: para que "lea" un PDF hay que extraer su texto antes de pasárselo.
// Extractores, en orden:
//   1. pdftotext (Poppler), si está instalado: Linux suele traerlo; en macOS `brew install poppler`.
//   2. macOS: PDFKit del sistema vía osascript (JavaScript for Automation), sin instalar nada.
// Los programas se ejecutan directamente (fork + execvp, sin shell), así que un nombre de archivo
// con caracteres especiales no puede inyectar comandos.
//
// Un PDF escaneado (solo imágenes) no tiene texto: se informa como error (requeriría OCR).

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

#if !defined(_WIN32)
#    include <csignal>
#    include <fcntl.h>
#    include <sys/wait.h>
#    include <unistd.h>
#endif

namespace nai {

// ¿El archivo es un PDF? Se decide por la firma "%PDF-" (no por la extensión).
inline bool es_pdf(const std::string & ruta) {
    std::ifstream f(ruta, std::ios::binary);
    char cab[5] = {0};
    return f.read(cab, sizeof(cab)) && std::string(cab, sizeof(cab)) == "%PDF-";
}

#if !defined(_WIN32)
// Ejecuta argv[0] con sus argumentos (sin shell) y captura su salida estándar, hasta `limite` bytes.
// Devuelve false si el programa no existe o no se pudo ejecutar; `codigo` es su código de salida.
inline bool ejecutar_y_capturar(const std::vector<std::string> & args, std::string & salida, size_t limite, int & codigo) {
    int tubo[2];
    if (args.empty() || pipe(tubo) != 0) {
        return false;
    }
    const pid_t pid = fork();
    if (pid < 0) {
        close(tubo[0]);
        close(tubo[1]);
        return false;
    }
    if (pid == 0) {
        dup2(tubo[1], STDOUT_FILENO);
        const int nulo = open("/dev/null", O_WRONLY);
        if (nulo >= 0) {
            dup2(nulo, STDERR_FILENO);
        }
        close(tubo[0]);
        close(tubo[1]);
        std::vector<char *> c_args;
        for (const auto & a : args) {
            c_args.push_back(const_cast<char *>(a.c_str()));
        }
        c_args.push_back(nullptr);
        execvp(c_args[0], c_args.data());
        _exit(127); // no existe el programa
    }
    close(tubo[1]);
    salida.clear();
    char buf[65536];
    bool cortado = false;
    ssize_t n;
    while ((n = read(tubo[0], buf, sizeof(buf))) > 0) {
        if (salida.size() + (size_t) n > limite) {
            salida.append(buf, limite - salida.size());
            cortado = true;
            kill(pid, SIGTERM);
            break;
        }
        salida.append(buf, (size_t) n);
    }
    close(tubo[0]);
    int estado = 0;
    waitpid(pid, &estado, 0);
    codigo = cortado ? 0 : (WIFEXITED(estado) ? WEXITSTATUS(estado) : -1);
    return codigo != 127;
}
#endif

// Extrae el texto de un PDF. Devuelve false con un mensaje en `error` si no se pudo.
inline bool pdf_a_texto(const std::string & ruta, std::string & texto, std::string & error,
                        size_t limite = 16u * 1024u * 1024u) {
#if defined(_WIN32)
    (void) ruta; (void) texto; (void) limite;
    error = "leer PDF no está soportado en Windows todavía; convierte el PDF a texto primero";
    return false;
#else
    int codigo = 0;
    bool algun_extractor = false;

    // 1. pdftotext (Poppler): -layout conserva columnas y tablas; "-" escribe en la salida estándar
    if (ejecutar_y_capturar({"pdftotext", "-layout", "-enc", "UTF-8", "-q", ruta, "-"}, texto, limite, codigo)) {
        algun_extractor = true;
        if (codigo == 0 && texto.find_first_not_of(" \t\r\n\f") != std::string::npos) {
            return true;
        }
    }

#    if defined(__APPLE__)
    // 2. macOS: PDFKit del sistema mediante JavaScript for Automation
    static const char * script =
        "function run(argv) {"
        "  ObjC.import('Quartz');"
        "  var d = $.PDFDocument.alloc.initWithURL($.NSURL.fileURLWithPath(argv[0]));"
        "  if (!d || d.isNil()) { return ''; }"
        "  var s = d.string;"
        "  return (s && !s.isNil()) ? s.js : '';"
        "}";
    if (ejecutar_y_capturar({"osascript", "-l", "JavaScript", "-e", script, ruta}, texto, limite, codigo)) {
        algun_extractor = true;
        if (codigo == 0 && texto.find_first_not_of(" \t\r\n\f") != std::string::npos) {
            return true;
        }
    }
#    endif

    if (!algun_extractor) {
        error = "no hay extractor de PDF: instala Poppler (macOS: brew install poppler; Linux: poppler-utils)";
    } else {
        error = "el PDF no tiene texto extraíble (¿está escaneado? haría falta OCR) o está dañado/protegido";
    }
    texto.clear();
    return false;
#endif
}

} // namespace nai
