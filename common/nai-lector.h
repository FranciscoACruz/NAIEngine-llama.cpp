// [NAI] Lector de documentos: convierte URLs y archivos en texto para que la memoria NAI aprenda.
//
//   texto, Markdown, CSV, JSON, código  -> directo (Latin-1 se convierte a UTF-8)
//   HTML / páginas web                  -> contenido principal (<article>/<main>), sin menús ni pies
//   PDF                                 -> pdftotext (poppler)              si está instalado
//   DOCX, DOC, RTF, ODT, PAGES          -> textutil (macOS) o pandoc        si está instalado
//
// URLs: solo http/https; por defecto se rechazan direcciones internas (localhost, red local,
// metadatos de nube) para que nadie use el servidor para leer la red privada. La IP validada
// se fija para la conexión (evita el cambio de DNS entre la validación y la descarga).
#pragma once

#include <cstddef>
#include <string>

namespace nai_lector {

struct resultado {
    bool        ok = false;
    std::string texto;
    std::string tipo;    // "texto", "html", "pdf", "docx", ...
    std::string error;
};

resultado leer_url(const std::string & url, bool permitir_red_local = false, size_t max_bytes = 16u << 20);
resultado leer_bytes(const std::string & nombre, const std::string & contenido, const std::string & content_type = "");
resultado leer_archivo(const std::string & ruta);

// Texto de una página: prefiere <article>/<main>; quita scripts, estilos, menús, pies y formularios.
std::string html_a_texto(const std::string & html);
// Quita líneas de menos de 6 palabras (restos de navegación). Se aplica a las páginas web.
std::string limpiar_web(const std::string & texto);

} // namespace nai_lector
