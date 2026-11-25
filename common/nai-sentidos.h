// [NAI] Sentidos: ojos (imágenes y cuadros de video) y oído (audio) para NAI.
//
// NAI guarda y razona en texto. Lo que entra por los sentidos se convierte en texto antes de llegar a
// la memoria, a la biblioteca o a un modelo principal que no ve ni oye:
//
//   imagen -> ojos  : un llama-server con modelo de visión (p. ej. Qwen3-VL-2B + --mmproj) la describe
//   audio  -> oído  : un llama-server con modelo de audio (p. ej. Qwen3-ASR) o whisper-server la transcribe
//   video  -> ffmpeg saca algunos cuadros (ojos) y la pista de audio (oído)
//
// Si el modelo principal ya ve u oye (--mmproj), la conversación le pasa el medio tal cual y los
// sentidos solo se usan para lo que se aprende (la biblioteca guarda texto).
#pragma once

#include <string>

namespace nai_sentidos {

struct config {
    std::string ojos;          // URL de un servidor compatible con OpenAI que ve (vacío = sin ojos)
    std::string oido;          // URL de un servidor que oye; si termina en /inference, se usa el formato de whisper-server
    std::string clave;         // Bearer para los sentidos (p. ej. la --api-key del propio servidor)
    int         cuadros = 6;   // cuadros que se toman de un video
};

void           configurar(const config & c);
const config & actual();

// "imagen", "audio", "video" o "" según la extensión, el Content-Type y los primeros bytes.
std::string tipo_medio(const std::string & nombre, const std::string & content_type, const std::string & bytes);
// MIME a partir de los primeros bytes (o del tipo, si no se reconoce).
std::string mime_de(const std::string & tipo, const std::string & bytes);

// Convierte el medio en texto. "pedido" (opcional) es lo que preguntó el usuario, para que la
// descripción se enfoque en eso. Devuelve false con un error legible si no hay sentido o falla.
bool describir(const std::string & tipo, const std::string & bytes, const std::string & mime,
               const std::string & pedido, std::string & texto, std::string & error);

bool ver (const std::string & bytes, const std::string & mime, const std::string & pedido, std::string & texto, std::string & error);
bool oir (const std::string & bytes, const std::string & mime, std::string & texto, std::string & error);
bool ver_video(const std::string & bytes, const std::string & mime, const std::string & pedido, std::string & texto, std::string & error);

} // namespace nai_sentidos
