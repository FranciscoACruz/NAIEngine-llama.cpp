#!/usr/bin/env bash
# [NAI] Oído y voz: hablar con NAI.
#   oído: graba hasta que te callas (sox) y transcribe en español (whisper.cpp)
#   NAI:  manda el texto al servidor con una sesión, así recuerda la plática
#   voz:  lee la respuesta en voz alta (say en macOS; espeak-ng en Linux)
#
# Requisitos (macOS):
#   brew install whisper-cpp sox
#   un modelo de Whisper en formato ggml, por ejemplo:
#     curl -L -o ggml-small.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin
#     (más preciso: ggml-large-v3-turbo-q5_0.bin, ~550 MB)
#   el servidor de NAI corriendo con --nai-memoria
#
# Uso:
#   bash tools/nai/nai-voz.sh -w ggml-small.bin [-p 8080] [-s mi-sesion] [-v Paulina] [-n 300]
#   -n: tokens máximos por respuesta (hablada, conviene que sea breve)
# Ctrl+C para salir.

WHISPER=""
PUERTO=8080
SESION="voz"
VOZ=""
MAX=300
while getopts "w:p:s:v:n:" o; do
  case $o in
    w) WHISPER="$OPTARG" ;;
    p) PUERTO="$OPTARG" ;;
    s) SESION="$OPTARG" ;;
    v) VOZ="$OPTARG" ;;
    n) MAX="$OPTARG" ;;
    *) exit 1 ;;
  esac
done
URL="http://localhost:$PUERTO"
TMP="${TMPDIR:-/tmp}/nai-voz-$$"
mkdir -p "$TMP"
trap 'rm -rf "$TMP"; echo; echo "Hasta luego."; exit 0' INT TERM

falla() { echo "ERROR: $1"; exit 1; }
[ -f "$WHISPER" ] || falla "indica el modelo de Whisper con -w (por ejemplo -w ggml-small.bin)"
command -v rec >/dev/null || falla "falta sox para grabar (brew install sox)"
command -v python3 >/dev/null || falla "falta python3"
STT=$(command -v whisper-cli || command -v whisper-cpp) || falla "falta whisper.cpp (brew install whisper-cpp)"
curl -s "$URL/health" | grep -q ok || falla "el servidor de NAI no responde en $URL"

# Voz: say (macOS) con una voz en español si existe; si no, espeak-ng (Linux)
if command -v say >/dev/null; then
  [ -n "$VOZ" ] || VOZ=$(say -v '?' 2>/dev/null | awk '/es_MX/ {print $1; exit}')
  [ -n "$VOZ" ] || VOZ=$(say -v '?' 2>/dev/null | awk '/es_ES/ {print $1; exit}')
  hablar() { if [ -n "$VOZ" ]; then say -v "$VOZ" -- "$1"; else say -- "$1"; fi; }
elif command -v espeak-ng >/dev/null; then
  hablar() { espeak-ng -v es-419 -- "$1"; }
else
  hablar() { :; }
  echo "(sin voz: no encontré say ni espeak-ng; solo verás el texto)"
fi

echo "Habla cuando quieras; NAI te escucha hasta que guardas silencio. Ctrl+C para salir."
while true; do
  echo
  echo "[escuchando...]"
  # empieza al detectar sonido y para tras 1.5 s de silencio; tope de 60 s
  rec -q -c 1 -r 16000 -b 16 "$TMP/voz.wav" silence 1 0.1 2% 1 1.5 2% trim 0 60 2>/dev/null || continue
  "$STT" -m "$WHISPER" -l es -nt -otxt -of "$TMP/voz" -f "$TMP/voz.wav" >/dev/null 2>&1 || continue
  TEXTO=$(tr '\n' ' ' < "$TMP/voz.txt" | sed 's/^ *//; s/ *$//')
  [ -n "$TEXTO" ] || continue
  echo "  Tú:  $TEXTO"
  RESP=$(python3 - "$URL" "$SESION" "$TEXTO" "$MAX" <<'EOF'
import json, sys, urllib.request
url, sesion, texto, maximo = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
cuerpo = {"nai_session": sesion, "max_tokens": maximo,
          "messages": [{"role": "user", "content": texto}]}
req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(cuerpo).encode(),
                             headers={"Content-Type": "application/json"})
try:
    d = json.load(urllib.request.urlopen(req, timeout=600))
    print(d["choices"][0]["message"]["content"].strip())
except Exception as e:
    print("No pude responder:", e)
EOF
)
  echo "  NAI: $RESP"
  # para la voz: sin marcas de formato (asteriscos, almohadillas, guiones de lista)
  hablar "$(printf '%s' "$RESP" | sed 's/[*#`_]//g; s/^- //')"
done
