#!/usr/bin/env bash
# [NAI] Arranca NAI completo con un solo comando (o ./nai.sh desde la raíz del repo): principal que ve,
# oído, cerebelo, hipocampo y memoria. Si falta el modelo de alguna parte, no arranca a medias: dice cuál falta.
# Si el oído o NAI se caen mientras trabajan, se vuelven a levantar solos (la memoria está en disco).
# Ctrl+C cierra todo. No cierra ningún otro programa: si un puerto está ocupado, avisa y se detiene.
#
#   bash tools/nai/nai-arrancar.sh            # todo: ojos (Qwen3-VL), oído (Qwen3-ASR), cerebelo, hipocampo
#   bash tools/nai/nai-arrancar.sh --sin-oido # sin oído
#   bash tools/nai/nai-arrancar.sh --sin-cerebelo --sin-hipocampo   # solo si decides prescindir de ellos
#   bash tools/nai/nai-arrancar.sh --whisper ~/nai-modelos/ggml-small.bin   # oído con whisper-server
#   bash tools/nai/nai-arrancar.sh --texto    # modelo de solo texto (Qwen3-1.7B), sin sentidos
#
# Todo se puede cambiar con variables de entorno, por ejemplo:
#   NAI_PUERTO=8090 NAI_MODELOS=/Volumes/Datos/modelos bash tools/nai/nai-arrancar.sh
#
# Variables: NAI_MODELOS (~/nai-modelos), NAI_DATOS (~/nai-datos), NAI_PUERTO (8080), NAI_PUERTO_OIDO (8082),
#            NAI_PRINCIPAL, NAI_PROYECTOR, NAI_OIDO_MODELO, NAI_OIDO_PROYECTOR, NAI_CEREBELO, NAI_HIPOCAMPO,
#            NAI_CONTEXTO (8192), NAI_EXTRA (flags adicionales para NAI)
set -u

RAIZ="$(cd "$(dirname "$0")/../.." && pwd)"
SERVIDOR="$RAIZ/build/bin/llama-server"
MODELOS="${NAI_MODELOS:-$HOME/nai-modelos}"
DATOS="${NAI_DATOS:-$HOME/nai-datos}"
PUERTO="${NAI_PUERTO:-8080}"
PUERTO_OIDO="${NAI_PUERTO_OIDO:-8082}"
CONTEXTO="${NAI_CONTEXTO:-8192}"
PRINCIPAL="${NAI_PRINCIPAL:-$MODELOS/Qwen3-VL-2B-Instruct-Q8_0.gguf}"
PROYECTOR="${NAI_PROYECTOR:-$MODELOS/mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf}"
OIDO_MODELO="${NAI_OIDO_MODELO:-$MODELOS/Qwen3-ASR-0.6B-Q8_0.gguf}"
OIDO_PROYECTOR="${NAI_OIDO_PROYECTOR:-$MODELOS/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf}"
CEREBELO="${NAI_CEREBELO:-$MODELOS/Qwen3-0.6B-Q8_0.gguf}"
HIPOCAMPO="${NAI_HIPOCAMPO:-$MODELOS/Qwen3-Embedding-0.6B-Q8_0.gguf}"
SIN_OIDO=0
WHISPER=""
TEXTO=0

while [ $# -gt 0 ]; do
  case "$1" in
    --sin-oido) SIN_OIDO=1 ;;
    --sin-cerebelo)  CEREBELO="" ;;
    --sin-hipocampo) HIPOCAMPO="" ;;
    --whisper)  WHISPER="${2:-}"; shift ;;
    --texto)    TEXTO=1 ;;
    -h|--help)  sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "Opción desconocida: $1 (usa --help)"; exit 1 ;;
  esac
  shift
done

if [ "$TEXTO" = 1 ]; then
  PRINCIPAL="${NAI_PRINCIPAL:-$MODELOS/Qwen3-1.7B-Q8_0.gguf}"
  PROYECTOR=""
  SIN_OIDO=1
fi

falta() { echo "✗ $1"; exit 1; }
puerto_ocupado() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }
esperar() {   # esperar URL PID nombre
  for _ in $(seq 1 180); do
    curl -s "$1/health" 2>/dev/null | grep -q '"ok"' && return 0
    kill -0 "$2" 2>/dev/null || return 1
    sleep 1
  done
  return 1
}

[ -x "$SERVIDOR" ] || falta "No está compilado: corre primero  cmake --build build -j4 --target llama-server  en $RAIZ"
[ -f "$PRINCIPAL" ] || falta "No encuentro el modelo principal: $PRINCIPAL"
[ -z "$PROYECTOR" ] || [ -f "$PROYECTOR" ] || falta "No encuentro el proyector de visión: $PROYECTOR"
puerto_ocupado "$PUERTO" && falta "El puerto $PUERTO está ocupado (revisa con: lsof -nP -iTCP:$PUERTO -sTCP:LISTEN). Usa otro con NAI_PUERTO=8090"
# todas las partes: si falta un modelo, no se arranca a medias (las --sin-... son decisión explícita)
[ -z "$CEREBELO" ]  || [ -f "$CEREBELO" ]  || falta "No encuentro el cerebelo: $CEREBELO (descárgalo, o arranca con --sin-cerebelo si decides no usarlo)"
[ -z "$HIPOCAMPO" ] || [ -f "$HIPOCAMPO" ] || falta "No encuentro el hipocampo: $HIPOCAMPO (descárgalo, o arranca con --sin-hipocampo si decides no usarlo)"
command -v ffmpeg >/dev/null 2>&1 || echo "! Sin ffmpeg: no podrá ver videos ni oír notas de voz que no sean WAV/MP3/FLAC (brew install ffmpeg)"

mkdir -p "$DATOS"
if [ ! -s "$HOME/.nai-sesion" ]; then
  (umask 077; openssl rand -hex 16 > "$HOME/.nai-sesion")
fi
SESION="$(cat "$HOME/.nai-sesion")"

# Guardián del oído: corre en segundo plano y, si el oído se cae, lo vuelve a levantar.
# Al cerrarse (TERM) cierra su servidor; si no responde en 5 s, lo fuerza. Solo toca lo que él arrancó.
guardian() {
  hijo=""
  terminar() {
    if [ -n "$hijo" ] && kill -0 "$hijo" 2>/dev/null; then
      kill "$hijo" 2>/dev/null
      for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$hijo" 2>/dev/null || break; sleep 0.5; done
      kill -0 "$hijo" 2>/dev/null && kill -9 "$hijo" 2>/dev/null
    fi
    exit 0
  }
  trap terminar TERM
  while :; do
    "$@" >> "$LOG_OIDO" 2>&1 &
    hijo=$!
    wait "$hijo"
    rc=$?
    [ "$rc" = 0 ] && exit 0   # se cerró en orden (Ctrl+C): no es una caída
    echo "! El oído se cerró (código $rc); se vuelve a levantar en 3 s" | tee -a "$LOG_OIDO"
    sleep 3
  done
}

PID_OIDO=""
PARAR=0
cerrar() {
  if [ -n "$PID_OIDO" ] && kill -0 "$PID_OIDO" 2>/dev/null; then
    kill "$PID_OIDO" 2>/dev/null
    for _ in $(seq 1 14); do kill -0 "$PID_OIDO" 2>/dev/null || break; sleep 0.5; done
    echo "· Oído cerrado"
  fi
}
trap cerrar EXIT
trap 'PARAR=1' INT TERM

URL_OIDO=""
if [ "$SIN_OIDO" = 0 ]; then
  puerto_ocupado "$PUERTO_OIDO" && falta "El puerto del oído $PUERTO_OIDO está ocupado. Usa otro con NAI_PUERTO_OIDO=8084"
  LOG_OIDO="$DATOS/oido.log"
  : > "$LOG_OIDO"
  if [ -n "$WHISPER" ]; then
    [ -f "$WHISPER" ] || falta "No encuentro el modelo de whisper: $WHISPER"
    command -v whisper-server >/dev/null 2>&1 || falta "No encuentro whisper-server (brew install whisper-cpp)"
    echo "· Arrancando el oído (whisper) en el puerto $PUERTO_OIDO..."
    guardian whisper-server -m "$WHISPER" -l es --port "$PUERTO_OIDO" --convert &
    PID_OIDO=$!
    URL_OIDO="http://127.0.0.1:$PUERTO_OIDO/inference"
    sleep 3
    kill -0 "$PID_OIDO" 2>/dev/null || falta "El oído no arrancó; revisa $LOG_OIDO"
  else
    [ -f "$OIDO_MODELO" ] || falta "No encuentro el modelo del oído: $OIDO_MODELO"
    [ -f "$OIDO_PROYECTOR" ] || falta "No encuentro el proyector del oído: $OIDO_PROYECTOR"
    echo "· Arrancando el oído (Qwen3-ASR) en el puerto $PUERTO_OIDO..."
    guardian "$SERVIDOR" -m "$OIDO_MODELO" --mmproj "$OIDO_PROYECTOR" --port "$PUERTO_OIDO" -fitt 1024 -c 4096 &
    PID_OIDO=$!
    URL_OIDO="http://127.0.0.1:$PUERTO_OIDO"
    esperar "$URL_OIDO" "$PID_OIDO" || falta "El oído no arrancó; revisa las últimas líneas de $LOG_OIDO"
  fi
  if grep -q "se vuelve a levantar" "$LOG_OIDO"; then falta "El oído se cae al arrancar; revisa $LOG_OIDO"; fi
  echo "· Oído listo ($URL_OIDO). Su registro: $LOG_OIDO"
fi

ARGS=(-m "$PRINCIPAL" --port "$PUERTO" -fitt 3072 -c "$CONTEXTO" --context-shift --nai-desalojo --reasoning off
      --nai-memoria --nai-sesiones "$DATOS" --nai-sesion-defecto "$SESION")
[ -n "$PROYECTOR" ] && ARGS+=(--mmproj "$PROYECTOR")
[ -n "$CEREBELO" ]  && ARGS+=(--nai-cerebelo "$CEREBELO")
[ -n "$HIPOCAMPO" ] && ARGS+=(--nai-hipocampo "$HIPOCAMPO")
[ -n "$URL_OIDO" ]  && ARGS+=(--nai-oido "$URL_OIDO")
# shellcheck disable=SC2206
[ -n "${NAI_EXTRA:-}" ] && ARGS+=(${NAI_EXTRA})

nombre() { if [ -n "$1" ]; then basename "$1"; else echo no; fi; }
echo "· Arrancando NAI en http://127.0.0.1:$PUERTO  (Ctrl+C para cerrar todo)"
echo "  principal: $(basename "$PRINCIPAL")${PROYECTOR:+ (ve)} | oído: ${URL_OIDO:-no} | cerebelo: $(nombre "$CEREBELO") | hipocampo: $(nombre "$HIPOCAMPO")"

# NAI se vuelve a levantar si se cae; si falla 3 veces seguidas al arrancar (menos de 60 s), algo está mal
# configurado y se detiene en vez de intentar para siempre.
fallas=0
while :; do
  inicio=$(date +%s)
  "$SERVIDOR" "${ARGS[@]}"
  rc=$?
  [ "$PARAR" = 1 ] || [ "$rc" = 0 ] && break   # Ctrl+C o cierre en orden: no es una caída
  if [ $(( $(date +%s) - inicio )) -lt 60 ]; then fallas=$((fallas + 1)); else fallas=0; fi
  [ "$fallas" -ge 3 ] && falta "NAI se cerró 3 veces al arrancar (código $rc). Revisa los mensajes de arriba"
  echo "! NAI se cerró (código $rc); se vuelve a levantar en 3 s. La memoria está guardada en $DATOS"
  sleep 3
  [ "$PARAR" = 1 ] && break
done
