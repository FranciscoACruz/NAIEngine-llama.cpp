// [NAI] Lector de documentos: convierte URLs y archivos en texto para que la memoria NAI aprenda.
//
//   texto, Markdown, CSV, JSON, código  -> directo (Latin-1 se convierte a UTF-8)
//   HTML / páginas web                  -> contenido principal (<article>/<main>), sin menús ni pies
//   PDF                                 -> pdftotext (poppler)              si está instalado
//   DOCX, DOC, RTF, ODT, PAGES          -> textutil (macOS) o pandoc        si está instalado
//   imágenes, audio, video              -> los sentidos de NAI (nai-sentidos.h): descripción o transcripción
//
// URLs: solo http/https; por defecto se rechazan direcciones internas (localhost, red local,
// metadatos de nube) para que nadie use el servidor para leer la red privada. La IP validada
// se fija para la conexión (evita el cambio de DNS entre la validación y la descarga).
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace nai_lector {

struct resultado {
    bool        ok = false;
    std::string texto;
    std::string tipo;    // "texto", "html", "pdf", "docx", ..., "imagen", "audio", "video"
    std::string error;
    std::string medio;   // imagen/audio/video: los bytes originales (para un modelo principal que ve u oye)
    std::string mime;
};

resultado leer_url(const std::string & url, bool permitir_red_local = false, size_t max_bytes = 48u << 20);
resultado leer_bytes(const std::string & nombre, const std::string & contenido, const std::string & content_type = "");
resultado leer_archivo(const std::string & ruta);

// Ejecuta un programa (sin shell) y devuelve su salida estándar. Usado también por los sentidos (ffmpeg).
bool ejecutar(const std::vector<std::string> & args, std::string & salida, std::string & error);
// Ruta nueva en la carpeta temporal del sistema: nai-lector-<aleatorio>.<ext>
std::string ruta_temporal(const std::string & ext);

// Texto de una página: prefiere <article>/<main>; quita scripts, estilos, menús, pies y formularios.
std::string html_a_texto(const std::string & html);
// Quita líneas de menos de 6 palabras (restos de navegación). Se aplica a las páginas web.
std::string limpiar_web(const std::string & texto);

} // namespace nai_lector
