# Motor NAI para llama.cpp

**Inercia latente, autopista de muestreo, memoria gravitatoria del KV cache, desalojo selectivo, sesiones persistentes de 24 h, memoria reconstructiva por regiones y un planificador de memoria entrelazado (en diseño).**

NAI es una extensión de [llama.cpp](https://github.com/ggml-org/llama.cpp) creada por **Ángelus Tenshi (Francisco Alejandro Cruz Castro)**. Parte de una idea: *gastar cómputo y memoria según cuánta información aporta cada token*, inspirada en principios físicos (inercia, gravedad, disipación). El marco teórico original está en [VISION.md](VISION.md). Este documento describe **lo que hace el código hoy**: cómo descargarlo, compilarlo, ejecutarlo, las fórmulas exactas que implementa, cómo medirlo y cómo modificarlo sin romper nada.

> **Autoría:** la idea, el marco teórico, los cálculos, la lógica y las reglas de negocio de NAI son de Ángelus Tenshi (Francisco Alejandro Cruz Castro). Ver [§14](#14-lógica-cálculos-y-reglas-de-negocio-del-autor) y [§15](#15-autoría-y-cómo-citar).

> **Principio de diseño:** todas las funciones NAI están **desactivadas por defecto**. Sin flags, el motor se comporta exactamente como llama.cpp. Cada función se activa con su propio flag y se puede medir por separado contra la versión sin ella.

## Índice

1. [Inicio rápido](#1-inicio-rápido)
2. [Descarga e instalación](#2-descarga-e-instalación)
3. [Ejecución](#3-ejecución)
4. [Funciones y flags](#4-funciones-y-flags)
5. [Fundamentos matemáticos](#5-fundamentos-matemáticos)
6. [Resultados medidos](#6-resultados-medidos)
7. [Cómo medir](#7-cómo-medir)
8. [Arquitectura del código](#8-arquitectura-del-código)
9. [Guía para desarrolladores](#9-guía-para-desarrolladores)
10. [Solución de problemas](#10-solución-de-problemas)
11. [Limitaciones y trabajo pendiente](#11-limitaciones-y-trabajo-pendiente)
12. [Trabajo relacionado](#12-trabajo-relacionado)
13. [Casos prácticos y campos de aplicación](#13-casos-prácticos-y-campos-de-aplicación)
14. [Lógica, cálculos y reglas de negocio del autor](#14-lógica-cálculos-y-reglas-de-negocio-del-autor)
15. [Autoría y cómo citar](#15-autoría-y-cómo-citar)

---

## 1. Inicio rápido

En macOS con Apple Silicon (en Linux cambia solo el paso de dependencias, ver §2):

```console
xcode-select --install
brew install cmake git
git clone https://github.com/FranciscoACruz/NAIEngine-llama.cpp.git
cd NAIEngine-llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4 --target llama-server llama-cli test-nai
./build/bin/test-nai
curl -L -o qwen2.5-0.5b-instruct-q4_k_m.gguf https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
./build/bin/llama-server -m qwen2.5-0.5b-instruct-q4_k_m.gguf --port 8080 --nai-sesiones ./sesiones --context-shift --nai-desalojo
```

Abre `http://localhost:8080` en el navegador o usa la API (§3.2).

---

## 2. Descarga e instalación

### 2.1 Requisitos

| Componente | Mínimo | Notas |
|---|---|---|
| Compilador C++17 | Clang 14, GCC 11, MSVC 2022 | En macOS viene con las Command Line Tools |
| CMake | 3.14 | `brew install cmake` / `apt install cmake` |
| Python 3 | 3.9 | Solo para las herramientas de `tools/nai/` y `crear_modelo_prueba.py` (`pip install numpy gguf`) |
| RAM | 4 GB | Modelo 0.5B Q4 ≈ 0.4 GB; 3B Q4 ≈ 2 GB. Con 8 GB compila con `-j 2` |
| Disco | ~2 GB | Código + compilación + un modelo pequeño |

El software no está atado a una arquitectura: el mismo código corre en CPU (x86, ARM), Metal (Apple), CUDA (NVIDIA), Vulkan y los demás backends de ggml. Las funciones NAI trabajan sobre los logits y sobre la máscara de atención, que existen en todos los backends.

### 2.2 Descargar el código

```console
git clone https://github.com/FranciscoACruz/NAIEngine-llama.cpp.git
cd NAIEngine-llama.cpp
```

Para actualizar después: `git pull`.

### 2.3 Dependencias por sistema

**macOS:**

```console
xcode-select --install
brew install cmake
```

**Ubuntu / Debian:**

```console
sudo apt update
sudo apt install -y build-essential cmake git python3-pip
```

**Windows:** instala Visual Studio 2022 con "Desarrollo para escritorio con C++" y CMake, y usa la "Developer PowerShell".

### 2.4 Compilar

Compilación estándar (detecta Metal en Apple Silicon automáticamente):

```console
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4 --target llama-server llama-cli test-nai nai_medir_capas nai_memoria
```

Variantes según el hardware:

| Hardware | Configuración |
|---|---|
| Apple Silicon (Metal) | `cmake -B build -DCMAKE_BUILD_TYPE=Release` |
| Solo CPU | `cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF` |
| NVIDIA | `cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON` |
| Vulkan (AMD, Intel, NVIDIA) | `cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON` |
| Equipos con poca RAM (≤ 8 GB) | Igual, pero compila con `-j 2` en lugar de `-j 4` |

Los binarios quedan en `build/bin/`. En Windows, en `build\bin\Release\`.

### 2.5 Verificar la instalación

```console
./build/bin/test-nai
```

No necesita modelo y debe terminar sin fallos. Cubre entropía, sorpresa, `exp_rapido`, log-sum-exp con NaN, predicciones pendientes, mapa de masa, ventana de desalojo, triaje, SipHash-2-4 (vectores oficiales), la tabla de sesiones, las claves y el troceo de la memoria, la lectura de HTML, el diccionario de tokens de la memoria (ida y vuelta y rechazo de datos dañados) y ChaCha20-Poly1305 (vector oficial del RFC 8439). También corre con `ctest --test-dir build -R test-nai --output-on-failure`.

### 2.6 Descargar un modelo

NAI usa modelos GGUF, igual que llama.cpp. Para empezar, Qwen2.5 Instruct en Q4_K_M:

```console
curl -L -o qwen2.5-0.5b-instruct-q4_k_m.gguf https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
curl -L -o qwen2.5-3b-instruct-q4_k_m.gguf https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
```

Para pruebas sin descargar nada hay un generador de modelos aleatorios diminutos (la salida no tiene sentido, pero ejercita todo el motor):

```console
pip3 install numpy gguf
python3 tests/nai-manual/crear_modelo_prueba.py /tmp/nai-prueba.gguf
```

Con `--capas 24 --embd 896 --cabezas 14 --cabezas-kv 2 --ff 4864 --f16 --vocab models/ggml-vocab-qwen2.gguf` crea un modelo aleatorio con la forma exacta de Qwen2.5-0.5B, útil para medir costos reales por token.

---

## 3. Ejecución

### 3.1 Servidor

```console
./build/bin/llama-server -m qwen2.5-0.5b-instruct-q4_k_m.gguf --port 8080 -c 8192 --context-shift --nai-desalojo --nai-sesiones ./sesiones --nai-sesion-ttl 86400
```

Expone la interfaz web en `http://localhost:8080` y una API compatible con OpenAI en `/v1/chat/completions`.

**Equipos con poca memoria (por ejemplo, Mac M1 de 8 GB).** `--fit` viene activado y ajusta capas y contexto a la memoria libre; deja más margen para el sistema con `-fitt 2048` (MiB libres por dispositivo) y compila con `-j4` para no saturar la RAM:

```console
cmake --build build --config Release -j4
./build/bin/llama-server -m qwen2.5-0.5b-instruct-q4_k_m.gguf --port 8080 -fitt 2048 --nai-sesiones ./sesiones --nai-memoria
```

### 3.2 API con sesión persistente

Cada petición puede llevar `"nai_session"`. El backend que llama al servidor genera el id (aleatorio, ≥ 128 bits) y lo asocia al usuario autenticado:

```console
SESION=$(openssl rand -hex 16)
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"nai_session\":\"$SESION\",\"messages\":[{\"role\":\"user\",\"content\":\"Hola, me llamo Ana\"}]}"
```

Si el servidor se reinicia o el slot se reutiliza, la siguiente petición con el mismo id restaura el KV cache del disco en lugar de reprocesar toda la conversación (§5.8).

### 3.3 Memoria reconstructiva y aprendizaje de documentos

```console
./build/bin/llama-server -m qwen2.5-3b-instruct-q4_k_m.gguf --port 8080 --nai-sesiones ./sesiones --nai-memoria
```

Con la memoria activa, el cliente manda **solo el mensaje nuevo** con su `nai_session`; el servidor agrega los recuerdos y los últimos 3 turnos. Si el cliente manda su historial completo, se respetan sus últimos turnos y el resto se sustituye por recuerdos. La interfaz web y los clientes OpenAI que no mandan `nai_session` usan la sesión `web` (`--nai-sesion-defecto ID`; vacío = sin memoria); las peticiones con `tools` no se tocan:

```console
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"nai_session\":\"$SESION\",\"messages\":[{\"role\":\"user\",\"content\":\"¿Cuánto ahorro cada quincena?\"}]}"
```

Aprender de un texto, una URL o un archivo:

```console
curl http://localhost:8080/nai/aprender -H "Content-Type: application/json" -d "{\"url\":\"https://ejemplo.org/articulo-sobre-la-milpa\",\"region\":\"agricultura\"}"
curl http://localhost:8080/nai/aprender -H "Content-Type: application/json" -d "{\"archivo\":\"$(base64 -i manual.pdf)\",\"nombre\":\"manual.pdf\"}"
curl http://localhost:8080/nai/memoria
```

Para PDF instala poppler (`brew install poppler`). Word, RTF y Pages se leen con `textutil`, que viene con macOS; en Linux, con `pandoc`.

#### Cómo pasarle información a NAI

El mejor formato es **Markdown o texto plano**, un archivo por tema, enviado con su región:

| Formato | Qué pasa al aprenderlo | ¿Conviene? |
|---|---|---|
| **Markdown / .txt** | Se lee directo y se parte por párrafos en trozos de hasta 600 caracteres; cada trozo es un recuerdo | **Sí, el mejor** |
| PDF | Depende de `pdftotext`: columnas, encabezados de página y palabras cortadas con guion se cuelan al texto | Solo si ya lo tienes así |
| Word (.docx) | `textutil` o `pandoc` lo convierten bien, con un paso más y posibles restos de formato | Aceptable |
| HTML | Se quitan etiquetas, menús y pies, pero la limpieza es heurística | Para páginas web, no para material propio |
| JSON | Se lee como texto tal cual: llaves, comillas y nombres de campo entran a la memoria y al entrenamiento como si fueran contenido | **No**. El JSON (JSONL) es la *salida* del reentrenamiento, no la entrada |

Cómo escribirlo para que NAI lo aprenda bien:

1. **Cada párrafo debe entenderse solo**, porque se vuelve un recuerdo independiente: «Rumi enseñaba que…» funciona; «Él decía que…» pierde a quién se refiere.
2. **Párrafos cortos**, de 2 a 4 oraciones (menos de 600 caracteres); si son más largos, el corte cae donde caiga.
3. **Un archivo por tema, con su región**, para no depender del clasificador:

```console
curl http://localhost:8080/nai/aprender -H "Content-Type: application/json" -d "{\"archivo\":\"$(base64 -i rumi.md)\",\"nombre\":\"rumi.md\",\"region\":\"espiritualidad\"}"
```

4. Para buscar material en la red: `docs/nai-fuentes.txt`, líneas `región|consulta o URL` (§3.6). Para lo más importante, quién es NAI y cómo entiende lo que aprende: la carta de identidad, `docs/IDENTIDAD_NAI.md` (§5.12), que se entrena antes que todo.

#### Qué hace NAI sola y qué depende de ti

Con cualquier método de entrada, NAI hace sola:

1. **Fragmenta**: parte el texto por párrafos, en trozos de hasta 600 caracteres.
2. **Juzga la naturaleza** (guardián, §5.9): conocimiento, sabiduría, desahogo o manipulación; un documento manipulador se descarta.
3. **Ubica la región**: si no se indica, la clasifica por predicción.
4. **Guarda y crea las claves**: codifica cada trozo y lo marca con sus palabras más sorprendentes, para encontrarlo después.
5. **Distribuye por importancia**: peso de la región × uso × naturaleza. Cuando la biblioteca se llena (8 MiB por defecto, unos 100 mil recuerdos), lo menos importante baja de nivel (exacto → idea → pista) antes de olvidarse.
6. **Relaciona**: al preguntar, trae lo que coincide y, aparte, un fragmento asociado como comentario opcional. Si la pregunta no toca la memoria, no trae nada.

Lo que no hace sola, o todavía no hace bien:

| Punto | Qué pasa hoy | Qué hacer |
|---|---|---|
| **Pasar lo aprendido al modelo** | La memoria funciona sola; el reentrenamiento (LoRA) se corre a mano con `nai_reentrenar.py` y se carga con `--lora` | Correrlo cuando haya suficiente material (§3.6) |
| **Juzgar la importancia al leer** | No entiende la importancia de un dato la primera vez; la aprende con el uso (lo consultado sube, lo que nadie toca baja) | Indicar la región del material que ya sabes que importa |
| **Clasificar con modelos chicos** | La precisión depende del modelo; con Qwen2.5-0.5B aún está por medirse con el método por predicción | Indicar `"region"` al enviar; medir con `nai_memoria` (§3.7) |
| **Relacionar por significado** | Solo con claves, «mascota» no encuentra un recuerdo sobre «perro» | Activar el hipocampo: `--nai-hipocampo` (§3.8) |
| **Recordar las conversaciones a largo plazo** | La biblioteca (documentos) no caduca. **La memoria de cada conversación caduca a las 24 h sin uso** (`--nai-sesion-ttl 86400`) | Para recordar pláticas por meses, súbelo, p. ej. `--nai-sesion-ttl 31536000` (un año) |

### 3.4 Línea de comandos

```console
./build/bin/llama-cli -m qwen2.5-0.5b-instruct-q4_k_m.gguf --nai-triaje --nai-autopista
```

### 3.5 Cifrado de sesiones y memorias

```console
mkdir -p ~/.nai && ./build/bin/llama-server -m qwen2.5-3b-instruct-q4_k_m.gguf --nai-sesiones ./sesiones --nai-memoria --nai-cifrado ~/.nai/clave.hex
```

La primera vez se crea la clave (permisos 0600). Respáldala: sin ella las conversaciones guardadas no se recuperan. No la guardes dentro de `./sesiones`.

### 3.5.1 Login: una entidad para la familia

```console
./build/bin/llama-server -m qwen2.5-3b-instruct-q4_k_m.gguf --nai-sesiones ./sesiones --nai-memoria \
  --nai-cifrado ~/.nai/clave.hex --nai-login --host 0.0.0.0
curl -s -X POST http://127.0.0.1:8080/nai/configurar -H 'Content-Type: application/json' \
  -d '{"usuario":"angelus","contrasena":"una-frase-larga-y-tuya"}'
curl -s -X POST http://127.0.0.1:8080/nai/login -H 'Content-Type: application/json' \
  -d '{"usuario":"angelus","contrasena":"una-frase-larga-y-tuya"}'
```

El dueño (creado solo desde la misma máquina) enseña a la entidad y crea cuentas de invitado para
parientes y amigos, que consultan lo aprendido desde cualquier dispositivo pero no pueden enseñar,
administrar, enviar código ni comandos. Lo sensible (PIN, NIP, CLABE, contraseñas, tarjetas,
saldos) queda privado automáticamente y solo lo ve el dueño. Guía completa, endpoints y
despliegue local, en casa o central: [docs/API_ACCESO.md](docs/API_ACCESO.md).

### 3.6 Reentrenamiento por currículo (Mac con Apple Silicon)

```console
pip3 install mlx-lm safetensors numpy
pip3 install -r requirements/requirements-convert_lora_to_gguf.txt
mkdir -p ~/searxng
printf 'use_default_settings: true\nserver:\n  secret_key: "%s"\n  limiter: false\nsearch:\n  formats:\n    - html\n    - json\n' "$(openssl rand -hex 32)" > ~/searxng/settings.yml
docker run -d --name searxng -p 8888:8080 -v ~/searxng:/etc/searxng searxng/searxng
python3 tools/nai/nai_reentrenar.py fuentes --servidor http://localhost:8080 --searxng http://localhost:8888 --esperar
python3 tools/nai/nai_reentrenar.py todo --modelo qwen2.5-0.5b-instruct-q4_k_m.gguf --biblioteca sesiones/biblioteca.mem --servidor http://localhost:8080 --modelo-mlx mlx-community/Qwen2.5-0.5B-Instruct-4bit --base-hf Qwen/Qwen2.5-0.5B-Instruct
./build/bin/llama-server -m qwen2.5-0.5b-instruct-q4_k_m.gguf --lora nai-reentreno/nai-adaptador.gguf --nai-sesiones ./sesiones --nai-memoria
```

Las líneas `mkdir`, `printf` y `docker run` levantan SearxNG en tu máquina (requiere Docker): crean su configuración con el formato JSON activado (`search.formats`), que el script necesita, y lo dejan en el puerto 8888. Si en lugar de eso usas Brave Search, crea una clave gratuita en su sitio y usa `--buscador brave --clave-brave TU_CLAVE`. El servidor de NAI debe estar corriendo con el modelo **base** y `--nai-memoria`. Si la biblioteca está cifrada, agrega `--clave ~/.nai/clave.hex` al paso `todo`. Cada paso (`exportar`, `datos`, `entrenar`, `medir`, `convertir`) se puede correr por separado; `--help` muestra las opciones.

### 3.7 Herramientas de medición (requieren modelo)

```console
./build/bin/nai_medir_capas -m qwen2.5-0.5b-instruct-q4_k_m.gguf -n 64
./build/bin/nai_memoria -m qwen2.5-3b-instruct-q4_k_m.gguf -c "lápiz"
python3 tools/nai/nai_bench.py -m qwen2.5-0.5b-instruct-q4_k_m.gguf --rondas 3 --reps 3
./build/bin/nai_memoria -m Qwen3-0.6B-Q8_0.gguf --hipocampo Qwen3-Embedding-0.6B-Q8_0.gguf   # calibra --nai-hipocampo-umbral
bash tools/nai/probar-nai.sh qwen2.5-0.5b-instruct-q4_k_m.gguf     # prueba completa: compilar, medir, conversar, aprender
```

### 3.8 Varios modelos: cerebelo, hipocampo, especialista, sentidos y voz

NAI puede repartir el trabajo como las partes de un cerebro (detalle en `docs/RUTA_INFORMACION.md`). Cada pieza es opcional; sin ellas, el modelo principal hace todo.

```console
./build/bin/llama-server -m Qwen3-1.7B-Q8_0.gguf --port 8080 -fitt 2048 --reasoning off \
  --nai-sesiones ./sesiones --nai-memoria --nai-sesion-ttl 31536000 \
  --nai-cerebelo Qwen3-0.6B-Q8_0.gguf \
  --nai-hipocampo Qwen3-Embedding-0.6B-Q8_0.gguf \
  --nai-especialista http://localhost:8081
```

| Pieza | Flag | Qué hace |
|---|---|---|
| Cerebelo | `--nai-cerebelo FILE` | Un modelo chico se encarga de la memoria, el guardián y la clasificación; el principal solo conversa. Para usarlo también como borrador especulativo, pásalo además con `-md` (debe ser de la misma familia que el principal) |
| Hipocampo | `--nai-hipocampo FILE` | Un modelo de embeddings relaciona por significado: «mi mascota» encuentra «mi perro» aunque no compartan palabras. Calibra el umbral con `nai_memoria --hipocampo` |
| Especialista | `--nai-especialista URL` | Las preguntas de código y ciberseguridad técnica van a otro `llama-server` (por ejemplo, Qwen2.5-Coder en el 8081). Si no está corriendo, responde el principal: se arranca solo cuando se necesita |
| Ojos | `--mmproj FILE` en el principal, o `--nai-ojos URL` | NAI ve imágenes y videos (sección 3.9) |
| Oído | `--nai-oido URL` | NAI oye audios, notas de voz y la pista de los videos (sección 3.9) |
| Voz (hablar con NAI) | `tools/nai/nai-voz.sh` | Habla con NAI: graba (sox), transcribe en español (whisper.cpp), envía con una sesión y lee la respuesta (`say` en macOS) |

`--reasoning off` apaga el modo de pensamiento de Qwen3 para que la plática sea fluida; una petición puede pedirlo con `"chat_template_kwargs": {"enable_thinking": true}` (por ejemplo, para lógica o matemáticas). Los modelos: [Qwen3-1.7B-GGUF](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF), [Qwen3-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF), [Qwen3-Embedding-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF).

El especialista se arranca aparte, cuando se necesite:

```console
./build/bin/llama-server -m Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf --port 8081 -fitt 2048
```

Para hablar con NAI (macOS): `brew install whisper-cpp sox`, descarga un modelo de Whisper (`ggml-small.bin` o `ggml-large-v3-turbo-q5_0.bin`) y ejecuta `bash tools/nai/nai-voz.sh -w ggml-small.bin`.

### 3.9 Sentidos: imágenes, audio y video

NAI piensa y recuerda en texto. Lo que entra por los sentidos se le pasa al modelo principal tal cual si él lo entiende (un modelo de visión o de audio con su `--mmproj`); si no, los **ojos** o el **oído** lo convierten en texto antes. Así responde, lo recuerda en la memoria y lo aprende en la biblioteca.

| Entra | Por dónde | Qué hace NAI |
|---|---|---|
| Imagen adjunta en el chat | `image_url` (la interfaz web la deja adjuntar) | El principal la ve, o los ojos la describen (texto visible incluido) |
| Audio o nota de voz | `input_audio` (WAV, MP3, FLAC; con ffmpeg también OGG, WEBM, M4A) | El principal la oye, o el oído la transcribe |
| Video | `input_video` (MP4, MOV, WEBM, MKV, AVI; requiere ffmpeg) | Toma `--nai-cuadros` cuadros para los ojos y la pista de audio para el oído |
| URL de imagen, audio o video en el texto | como cualquier URL de la plática | La abre, se la pasa al principal si la entiende y, si escribe el dueño, aprende su descripción |
| Archivo o URL en `POST /nai/aprender` | `{"nombre": "foto.jpg", "archivo": "<base64>"}` o `{"url": ...}` | La biblioteca guarda la descripción o la transcripción |

Configuración recomendada para una Mac de 8 GB: el principal ve por sí mismo y el oído va aparte.

```console
./build/bin/llama-server -m Qwen3-VL-2B-Instruct-Q8_0.gguf --mmproj mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf \
  --port 8080 -fitt 3072 -c 8192 --context-shift --nai-desalojo \
  --nai-memoria --nai-sesiones ~/nai-datos --nai-oido http://127.0.0.1:8082

./build/bin/llama-server -m Qwen3-ASR-0.6B-Q8_0.gguf --mmproj mmproj-Qwen3-ASR-0.6B-Q8_0.gguf --port 8082 -fitt 1024
```

- **Ojos.** Si el principal ya ve (`--mmproj`), los ojos son el mismo servidor; no hace falta `--nai-ojos`. Con un principal que solo lee texto, `--nai-ojos http://127.0.0.1:8083` apunta a otro `llama-server` con un modelo de visión.
- **Oído.** `--nai-oido` acepta un `llama-server` con un modelo de audio (Qwen3-ASR, Voxtral, Ultravox) o un `whisper-server` de whisper.cpp con la URL terminada en `/inference`. El audio en llama.cpp aún es experimental; whisper es el respaldo.
- **ffmpeg** (`brew install ffmpeg`) hace falta para video y para audio que no sea WAV, MP3 o FLAC.
- Las peticiones de NAI a sus propios sentidos llevan la cabecera `X-NAI-Sentido`: no pasan por la memoria, las URL ni el especialista. Con `--nai-login`, el principal no se puede describir a sí mismo para la biblioteca: usa `--nai-ojos`/`--nai-oido` con otro servidor.
- Lo que describen los ojos (por ejemplo, el texto de un cartel en una foto) llega como datos, no como instrucciones, y pasa por el guardián antes de aprenderse.
- Una misma imagen, audio o video se describe una sola vez aunque la interfaz lo reenvíe en cada turno.

Modelos: [Qwen3-VL-2B-Instruct-GGUF](https://huggingface.co/ggml-org/Qwen3-VL-2B-Instruct-GGUF) (1.8 GB más un proyector de 445 MB) y [Qwen3-ASR-0.6B-GGUF](https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF) (805 MB más 214 MB).

---

## 4. Funciones y flags

| Función | Flag | Por defecto | ¿Cambia la salida? | Qué hace |
|---|---|---|---|---|
| Autopista de muestreo | `--nai-autopista` | apagada | **No** con temp 0. Con temp > 0, error acotado por `1 − p_min` | Devuelve el top-1 sin recorrer la cadena de samplers cuando el resultado ya está decidido |
| Umbral de la autopista | `--nai-autopista-p-min P` | `0.99` | — | Probabilidad mínima del top-1 para tomar el atajo con temp > 0. Rango (0, 1] |
| Desalojo selectivo | `--context-shift --nai-desalojo` | apagado | Solo cuando el contexto se llena | En el *context shift* descarta la ventana de **menor masa** en lugar de la más antigua, y al recortar un historial que no cabe elige qué olvidar por masa |
| Recorte de historial | siempre activo | — | Solo cuando no cabe | Si lo que llega al modelo (historial del cliente, o recuerdos + turnos recientes con `--nai-memoria`, o un mensaje enorme) no cabe en el contexto, se recorta (inicio + lo más reciente) en lugar de rechazar la petición; la respuesta lo marca con `"truncated": true`. Ninguna conversación se bloquea por tamaño, con o sin flags |
| Lectura de documentos en CLI y herramientas | `/read archivo.pdf`, `--tools read_file` | — | No | Extrae el texto de PDF (`pdftotext` o, en macOS, PDFKit del sistema), Word, RTF y ODT en lugar de pasar sus bytes; un PDF escaneado sin texto se rechaza. Un solo lector para todo: `common/nai-lector.cpp` |
| Memoria gravitatoria *(experimental)* | `--nai-gravedad γ` | `0` (apagada) | **Sí** | Atenúa la atención hacia tokens generados de poca masa: sesgo `−γ·Δt/(masa+ε)` |
| Suavizado de la gravedad | `--nai-gravedad-eps ε` | `0.05` | — | Evita dividir entre cero en masas muy bajas. Debe ser > 0 |
| Sesiones persistentes | `--nai-sesiones DIR` | apagadas | No | Guarda y restaura el KV cache de cada conversación |
| Caducidad de sesiones | `--nai-sesion-ttl SEG` | `86400` (24 h) | — | Segundos sin actividad tras los que la sesión y su archivo se eliminan |
| Triaje (CLI) | `--nai-triaje` | apagado | Solo pide aclaración | Detecta peticiones cortas y ambiguas y pregunta antes de generar |
| Memoria reconstructiva | `--nai-memoria` (requiere `--nai-sesiones DIR`) | apagada | Cambia lo que ve el modelo: recuerdos en lugar del historial | La conversación no se acumula en el KV cache; se guarda comprimida con el modelo en una memoria fija por regiones |
| Tamaño de la memoria por sesión | `--nai-memoria-bytes N` | `16384` | — | Presupuesto fijo de cada sesión |
| Sesión por defecto | `--nai-sesion-defecto ID` | vacío (sin memoria) | — | Sesión de memoria para clientes sin `nai_session` (interfaz web, clientes OpenAI). Antes era `web` y todos esos clientes compartían memoria. Usa un id largo y secreto; con `--nai-login` no se usa |
| Login y roles | `--nai-login` (requiere `--nai-sesiones DIR`) | apagado | No | La API exige token. Un dueño enseña y administra; los invitados solo conversan y consultan. Ver [docs/API_ACCESO.md](docs/API_ACCESO.md) |
| Vigencia del token | `--nai-token-dias N` | `30` | — | Días sin uso antes de vencer; al usarlo se renueva |
| Tamaño de cada tarea de aprendizaje | `--nai-parte N` | `8000` | — | Un texto más largo se aprende en partes de N caracteres, guardadas en `DIR/cola` y procesadas una a una |
| Límite de la cola de aprendizaje | `--nai-cola N` | `8388608` | — | Caracteres pendientes máximos; más devuelve 429 |
| Cerebelo | `--nai-cerebelo FILE` | apagado (el principal hace todo) | No | Modelo chico para la memoria, el guardián y la clasificación (§3.8) |
| Hipocampo | `--nai-hipocampo FILE` | apagado (solo claves) | Agrega recuerdos por significado | Modelo de embeddings; cada recuerdo guarda una firma de 32 B (§5.9) |
| Umbral del hipocampo | `--nai-hipocampo-umbral F` | `0.5` | — | Similitud mínima para traer un recuerdo sin palabras en común; calíbralo con `nai_memoria --hipocampo` |
| URL en la plática | `--nai-sin-urls` para apagarlo | siempre encendido, desde cualquier cliente (interfaz web incluida), con o sin sesión y aun sin `--nai-memoria` | Lee las páginas (HTML, PDF, texto) | Si el mensaje trae una URL (hasta 2), NAI la abre y responde con su contenido (el principio, como datos y no como instrucciones). Con `--nai-memoria`, si quien escribe es el dueño, la aprende completa en la biblioteca, una vez por URL; un invitado solo la usa para su respuesta. Una página con manipulación no se usa ni se aprende; una que no abre, se le dice al usuario |
| Anti-bucle | `--nai-anti-bucle N` | `8` (`0` = apagado) | Corta respuestas degeneradas | Si la salida repite el mismo fragmento N veces seguidas (bucle de repetición de modelos chicos con respuestas largas), se detiene. El guardián tampoco guarda ni reentrena texto en bucle |
| Ojos | `--nai-ojos URL` | el propio servidor si el principal tiene `--mmproj` de visión | Ve imágenes y videos | Describe lo que el principal no ve, y lo que se aprende en la biblioteca |
| Oído | `--nai-oido URL` | el propio servidor si el principal oye | Oye audio | `llama-server` con modelo de audio, o `whisper-server` (URL terminada en `/inference`) |
| Cuadros de video | `--nai-cuadros N` | 6 | Video | Cuadros que se toman de cada video para los ojos (requiere ffmpeg) |
| Especialista | `--nai-especialista URL` | apagado | Responde otro modelo | Las preguntas de código van a otro `llama-server`; si no responde, contesta el principal |
| Memoria compacta | `--nai-memoria-compacta` | apagada (portable) | — | Codifica los recuerdos con el modelo: ocupan menos, pero solo se leen con el mismo modelo, backend, `-ngl`, hilos y versión |
| Tamaño de la biblioteca | `--nai-biblioteca-bytes N` | `8388608` (8 MiB) | — | Presupuesto de lo aprendido de documentos |
| Regiones de la memoria | `--nai-regiones ARCHIVO` | 26 integradas | — | Líneas `peso\|tema\|semilla` |
| URLs de la red local | `--nai-lector-red-local` | rechazadas | — | Permite que `/nai/aprender` lea localhost o la red interna |
| Cifrado en disco | `--nai-cifrado ARCHIVO` | sin cifrado | No | Cifra sesiones (KV) y memorias (`.mem`) con ChaCha20-Poly1305; la clave vive en ARCHIVO, fuera de la carpeta de sesiones |
| Traza del planificador | `--nai-traza PREFIJO` | apagada | No | Escribe CSV por paso y por petición para calibrar el simulador del planificador |
| Auto-especulación por capas | `-md borrador.gguf` | — | **No** | Borrador = primeras K capas del modelo, verificado por el modelo completo |

> Verifica el valor por defecto exacto de cada flag con `./build/bin/llama-server --help | grep nai`.

---

## 5. Fundamentos matemáticos

Notación: $V$ es el vocabulario, $z \in \mathbb{R}^{|V|}$ los logits de una posición, $p = \mathrm{softmax}(z)$, $x_i$ el token en la posición $i$, $L$ el número de capas.

### 5.1 Masa semántica (`src/inercia_latente.hpp`)

La masa de un token es la **sorpresa normalizada** del token que realmente quedó en esa posición:

$$I_i = \frac{-\ln p(x_i \mid x_{\lt i})}{\ln |V|} = \frac{\mathrm{LSE}(z^{(i-1)}) - z^{(i-1)}_{x_i}}{\ln |V|} \in [0, 1]$$

donde $\mathrm{LSE}(z) = \ln \sum_v e^{z_v}$. El resultado se recorta a $[0, 1]$.

- **Masa baja:** el token era predecible (relleno, o algo que ya está en el contexto). Es candidato a desalojo o atenuación.
- **Masa alta:** el token aportó información nueva y se conserva.
- **Protegida ($I = 1$, `MASA_PROTEGIDA`):** posiciones sin predicción medida: el prompt y el mensaje de sistema. Nunca se desalojan ni se atenúan antes que el texto generado.

**Predicciones pendientes.** Los logits de la posición $i-1$ predicen el token $i$, que todavía no existe cuando se calculan. Por eso, tras cada `decode`, el contexto guarda $\mathrm{LSE}(z^{(i-1)})$ y el vector de logits necesario como *predicción pendiente* de la secuencia, y cuando llega el token $x_i$ calcula $I_i$ y la escribe **en la posición $i$**. Guardarla en $i-1$ (el error de versiones anteriores) asigna la sorpresa al token equivocado.

Corresponde al término $\mathcal{H}_{\text{local}}$ de VISION.md medido sobre el token mismo. El factor $\|W_K x_i\|_2$ de la fórmula original está pendiente (§11).

### 5.2 Log-sum-exp rápido y su cota de error

Calcular $\mathrm{LSE}$ sobre 152 000 logits por token es el costo principal del mapa de masa. NAI usa dos aproximaciones con error acotado:

**Corte a $C = 16$ nats** (`NAI_LSE_CORTE`). Con $m = \max_v z_v$:

$$\mathrm{LSE}(z) = m + \ln \sum_v e^{z_v - m} \approx m + \ln \sum_{z_v - m \gt  -C} e^{z_v - m}$$

Cada término omitido vale menos de $e^{-16} \approx 1.1 \times 10^{-7}$ y la suma conservada es $\ge 1$ (incluye $e^0$), así que el error es como máximo $\ln(1 + |V| e^{-C})$. Con $|V| = 152\,000$ eso es $\le 0.017$ nats, es decir $\le 0.0015$ en masa normalizada ($\div \ln|V| \approx 11.9$). En la práctica, con logits con pico, el error es $\lt  10^{-4}$ nats.

**`exp_rapido` sin ramas.** Para $x \le 0$:

$$e^x = 2^{t}, \quad t = x \log_2 e, \quad 2^t = 2^{n} \cdot 2^{f}, \quad n = \lfloor t \rfloor,\ f = t - n \in [0, 1)$$

$2^n$ se arma escribiendo directamente el exponente IEEE-754 ($(n + 127) \ll 23$) y $2^f$ se evalúa con un polinomio minimax de grado 5 por Horner. Error relativo $\lt  2 \times 10^{-6}$ en $[-16, 0]$ (lo verifica `test-nai`). Como no tiene ramas ni llamadas, el compilador lo vectoriza (NEON en Apple Silicon, AVX en x86).

**16 acumuladores.** El máximo y la suma se calculan con 16 acumuladores independientes para romper la dependencia entre iteraciones y dejar que el compilador use registros vectoriales. Los NaN sueltos se ignoran; si todo es NaN o $-\infty$, la masa queda protegida.

**Costo:** 0.348 → 0.171 ms por token con $|V| = 152$k (ver [BENCHMARKS.md](BENCHMARKS.md)).

### 5.3 Desalojo selectivo (`--nai-desalojo`)

Cuando el contexto de un slot se llena, llama.cpp descarta los $n_d$ tokens que siguen a los $n_{\text{keep}}$ protegidos (los más antiguos). NAI elige en cambio la ventana contigua de **menor masa total**:

$$p^* = \arg\min_{p \in [n_{\text{keep}},\ p_{\text{end}} - n_r - n_d]} \ \sum_{j=p}^{p+n_d-1} I_j$$

- $n_r = \min(256, n_{\text{left}}/4)$ tokens más recientes quedan fuera de la búsqueda: el modelo los necesita para continuar la frase.
- Se resuelve en $O(n)$ con una **suma deslizante**: $S(p+1) = S(p) + I_{p+n_d} - I_p$.
- Después: `seq_rm(p*, p*+n_d)`, `seq_add(p*+n_d, p_end, −n_d)` (RoPE desplaza las claves restantes) y el mapa de masa se recorre igual (`quitar_y_recorrer`).
- Si el mapa no está activo o no hay ventana válida, se usa el comportamiento original.

Como el prompt tiene masa protegida, el desalojo prefiere texto generado predecible; la prueba de aguja de `nai_bench.py` verifica que un dato al inicio sobreviva a varios *context shift*.

### 5.4 Memoria gravitatoria (`--nai-gravedad`, `src/llama-kv-cache.cpp`)

La "Fórmula Maestra" de VISION.md:

$$K_i(t+dt) = K_i(t)\cdot\exp\!\left(-\frac{\gamma\,dt}{I_i+\varepsilon}\right)$$

se aplica sobre el **peso de atención**, no sobre el vector $K$. Escalar $K$ no apaga un token: solo lleva su logit $q \cdot k$ hacia 0, y si los demás logits son negativos el token incluso gana peso.

**Derivación.** El peso de atención de la consulta $i$ sobre la clave $j$ es $w_{ij} = e^{s_{ij}} / \sum_k e^{s_{ik}}$ con $s_{ij} = q_i \cdot k_j / \sqrt{d}$. Multiplicar el peso no normalizado por un factor $f_{ij} = \exp(-\gamma\,\Delta t_{ij} / (I_j + \varepsilon))$ y renormalizar es exactamente:

$$w'_{ij} = \frac{e^{s_{ij}} f_{ij}}{\sum_k e^{s_{ik}} f_{ik}} = \mathrm{softmax}_j\!\left(s_{ij} + \ln f_{ij}\right), \qquad b_{ij} = \ln f_{ij} = -\frac{\gamma\,(p_i - p_j)}{I_j + \varepsilon}$$

Es decir, basta **sumar $b_{ij}$ a la máscara KQ** antes del softmax, con $\Delta t = p_i - p_j$ (distancia en posiciones).

- Funciona con y sin flash attention y en cualquier backend, porque solo toca la máscara.
- Los tokens protegidos ($I = 1$) decaen mucho más lento; con $\varepsilon$ pequeño su pozo es prácticamente indefinido.
- La "expansión de Hubble" de VISION.md §11.5 con corrimiento lineal en la distancia es este mismo término.
- **Experimental:** cambia la salida del modelo. Valores razonables de $\gamma$: `1e-4` a `1e-3`. No aplica a modelos con ALiBi, M-RoPE ni memoria recurrente o híbrida.
- **Escala:** con $\gamma = 5 \times 10^{-4}$ e $I = 0.05$, un token a 1000 posiciones recibe $b = -0.5/0.1 = -5$, es decir su peso se multiplica por $e^{-5} \approx 0.007$.

### 5.5 Autopista de muestreo (`--nai-autopista`, `common/sampling.cpp`)

Recorrer la cadena completa de samplers (top-k, top-p, min-p, temperatura, ordenamientos) cuesta $O(|V| \log |V|)$ en el peor caso. Si el resultado ya está decidido, NAI lo devuelve en una sola pasada $O(|V|)$.

- **Condición de seguridad:** la autopista solo se habilita si la cadena **no** contiene samplers que puedan cambiar el argmax: gramática, `logit_bias`, penalizaciones, DRY, XTC, typical, mirostat o dynatemp.
- **temp ≤ 0:** devuelve el top-1. Es idéntico a la cadena completa.
- **temp > 0:** toma el atajo solo si $p(\text{top-1}) \ge p_{\min}$.

**Cota de error.** Sea $P$ la distribución que la cadena completa muestrearía y $\delta_{\text{top}}$ la elección determinista del top-1. La distancia de variación total es:

$$\mathrm{TV}(P, \delta_{\text{top}}) = 1 - P(\text{top-1}) \le 1 - p_{\min}$$

Con el valor por defecto, como máximo 1 de cada 100 tokens muestreados podría diferir.

**Medición honesta:** en M1 la cadena de llama.cpp ya es muy eficiente y la autopista no mostró ganancia distinguible del ruido (ver §6). Se conserva porque es exacta y su costo es despreciable.

### 5.6 Inercia latente y salida temprana (`nai_medir_capas`)

$$\Delta h^{(k)} = 1 - \frac{\mathcal{H}\big(\mathrm{softmax}(z^{(k)})\big)}{\ln |V|}, \qquad z^{(k)} = W_{\text{out}} \cdot \mathrm{norm}(h^{(k)})$$

$z^{(k)}$ son los logits que daría el modelo si saliera en la capa $k$ (*logit lens*, con la cabeza de salida real `output.weight` del modelo). `nai_medir_capas` lee el estado de cada capa con el callback de evaluación de ggml (no modifica el motor) y mide por token:

- $1 - \cos(h_k, h_{k-1})$: cuánto cambia el estado entre capas;
- $\cos(h_k, h_L)$: qué tan cerca está ya del estado final;
- la **capa de resolución**: la primera desde la que la lente coincide con la predicción final y ya no cambia;
- para varios umbrales $\tau$, la regla "salir en la primera capa con $\Delta h \gt  \tau$": cuántos tokens saldrían antes, cuántos acertarían y qué fracción de capas se ahorraría.

**Por qué no se aplica un early exit directo:** si la regla se equivoca cambia la salida, y las capas omitidas no escriben K/V, así que los tokens siguientes atienden a un caché incompleto. La aplicación sin pérdida es la auto-especulación (§5.7).

### 5.7 Auto-especulación por capas

`tools/nai/nai_borrador_capas.py` crea un GGUF con las primeras $K$ capas del modelo y su cabeza de salida: es exactamente "salir en la capa $K$". El borrador propone $\gamma_d$ tokens y el modelo completo los verifica en un solo paso; la salida es idéntica a la del modelo completo.

Con tasa de aceptación $\alpha$ por token, el número esperado de tokens por ciclo es (Leviathan et al., 2023):

$$E[\text{tokens}] = \frac{1 - \alpha^{\gamma_d + 1}}{1 - \alpha}$$

y la aceleración respecto a decodificar normal es:

$$\text{aceleración} = \frac{E[\text{tokens}]}{\gamma_d \cdot c + T_{\text{ver}}}, \qquad c = \frac{K}{L},\quad T_{\text{ver}} = \frac{\text{costo de verificar } \gamma_d + 1 \text{ tokens}}{\text{costo de 1 token}}$$

**Medición en M1 8 GB:** verificar un lote de 8 tokens costó entre 2 y 3.4 pasos normales ($T_{\text{ver}} \approx 2$–$3.4$, porque en CPU/Metal pequeño el lote no es gratis como en una GPU grande), y la lente mostró que en Qwen2.5-0.5B casi ningún token se resuelve antes de la capa 21 de 24. Con esos números la aceleración queda $\lt  1$: **en ese equipo la auto-especulación es más lenta.** En GPUs donde $T_{\text{ver}} \approx 1$ y con modelos entrenados para salida temprana (LayerSkip) puede ganar. Mide siempre con `nai_medir_capas` antes de usarla.

### 5.8 Sesiones persistentes (`--nai-sesiones`, `src/nai-session-table.hpp`)

**Ciclo de vida:**

1. La petición trae `nai_session = id`.
2. Si la sesión existe en la tabla (o en disco) y no caducó, se carga su KV cache con `llama_state_seq_load_file` en el slot y solo se procesan los tokens nuevos.
3. Cuando el slot se libera, se guarda con `llama_state_seq_save_file`.
4. Tras `--nai-sesion-ttl` segundos sin actividad, la sesión y su archivo se eliminan. Al arrancar, el servidor recorre `DIR` y borra los archivos vencidos por fecha de modificación.

**Hash con clave.** El nombre de archivo es $h = \text{SipHash-2-4}_k(\text{id})$ en hexadecimal, con una clave $k$ de 128 bits aleatoria guardada en `DIR/.nai-clave` (permisos 0600). Sin la clave nadie puede fabricar un id que choque con el de otra sesión, y los logs solo muestran una etiqueta derivada de $h$, nunca el id.

**Tabla en memoria.** Direccionamiento abierto con *hashing de Fibonacci*:

$$\text{índice}(h) = \left(h \cdot \left\lfloor 2^{64}/\varphi \right\rfloor \bmod 2^{64}\right) \gg (64 - b), \qquad \lfloor 2^{64}/\varphi \rfloor = 11400714819323198485$$

donde $2^b$ es el tamaño de la tabla. Multiplicar por la razón áurea dispersa bien incluso claves con patrones. Los borrados dejan *lápidas* (tombstones) para no romper las cadenas de sondeo, y la tabla se reconstruye cuando $(\text{ocupadas} + \text{lápidas}) / \text{tamaño} \gt  0.7$.

**Costo en disco.** El KV cache por token es:

$$B_{KV} = n_{\text{capas}} \cdot 2 \cdot n_{\text{cabezas KV}} \cdot d_{\text{cabeza}} \cdot b_{\text{elemento}}$$

| Modelo | Capas | Cabezas KV | $d$ | Bytes por token (F16) | 4 000 tokens |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 24 | 2 | 64 | 12 288 | 49 MB |
| Qwen2.5-3B | 36 | 2 | 128 | 36 864 | 147 MB |

**Seguridad:**
- **El id es un secreto:** quien lo conozca puede recuperar la conversación. Genera ≥ 128 bits aleatorios (32 caracteres hex) en tu backend y lígalos al usuario autenticado. El servidor avisa si el id tiene menos de 16 caracteres.
- **Permisos:** carpeta 0700, archivos y clave 0600.
- **Cifrado:** sin `--nai-cifrado`, los archivos contienen la conversación sin cifrar. Con datos sensibles (por ejemplo, financieros) activa `--nai-cifrado` (§5.13) y, además, guarda la carpeta en un volumen cifrado (FileVault, LUKS, BitLocker).
- **Limitación:** no se restauran sesiones con multimodal ni con decodificación especulativa.

**Relación con `bloque_id`:** el campo `bloque_id` de la sesión está reservado para conectar NAI con un proyecto padre (generador de API de alto rendimiento) sin cambiar el formato.

### 5.9 Memoria reconstructiva por regiones (`--nai-memoria`, `common/nai-memoria.cpp`)

Memoria de **tamaño fijo** inspirada en la mente humana: los pesos del modelo son la "corteza" (conocimiento general, no cambia) y cada sesión tiene un "hipocampo" (tamaño fijo, organizado por regiones temáticas). Con `--nai-memoria`, **la conversación ya no se acumula en el KV cache**: el contexto de cada petición lleva solo el mensaje actual, el turno anterior y los recuerdos relevantes reconstruidos para esa pregunta. El número de tokens deja de ser el límite: una conversación de meses ocupa siempre el mismo espacio.

**Flujo en el servidor**

```
petición {nai_session, messages:[...]}            ← el cliente puede mandar solo el mensaje nuevo
   │ 1. claves de la pregunta → recuerdos de la sesión y de la biblioteca (GPS)
   │    sin coincidencias → no se trae nada: la memoria solo se consulta cuando la pregunta la toca
   │ 2. reconstruir con el modelo los recuerdos elegidos (máx. 2 por región: cruza temas)
   │    segundo salto: las claves de los mejores recuerdos activan otro (≥ 2 claves en común),
   │    que se ofrece aparte como comentario opcional («relacionado, menciónalo en una frase si aporta»)
   │ 3. mensajes = sistema + recuerdos · últimos 3 turnos · mensaje actual
   ▼
modelo responde (KV pequeño, se libera al terminar)
   │ 4. en segundo plano: clasificar el turno, codificar usuario y asistente,
   │    ajustar al presupuesto (bajar de nivel lo menos importante), guardar DIR/nai-<hash>.mem
```

**Dos codificaciones.** Por defecto, cada recuerdo se guarda con el **diccionario de tokens**: los IDs del vocabulario en enteros de longitud variable (7 bits por byte; un ID < 16384 ocupa 2 B). No hay números flotantes, así que se lee idéntico en cualquier máquina, backend, `-ngl`, número de hilos o versión de llama.cpp, y leerlo es una consulta de tabla: no pasa por el modelo. El modelo solo se usa al guardar, para medir la sorpresa (claves y pistas). Con `--nai-memoria-compacta` se usa la codificación guiada por el modelo que se describe abajo: ocupa menos con un modelo bien entrenado, pero los logits deben coincidir bit a bit, de modo que cambiar de backend o de `-ngl` marca los recuerdos como dañados. El bit 5 del byte de nivel indica la codificación de cada recuerdo, así que una misma memoria puede mezclar ambas y los `.mem` anteriores se siguen leyendo.

**Codificación exacta guiada por el modelo (`--nai-memoria-compacta`).** Cada recuerdo se guarda como el resultado de codificación aritmética (rANS de 64 bits, frecuencias de 24 bits) con las probabilidades del propio modelo, condicionadas por un prefijo gratuito que no se guarda ("Tema: física. Usuario:"). Por el teorema de Shannon, el tamaño se acerca a:

$$\text{bits}(x_{1..n}) \approx \sum_{i=1}^{n} -\log_2 p(x_i \mid x_{\lt i}, \text{tema}) = \ln|V| \cdot \log_2 e \cdot \sum_i I_i$$

Es decir, **el costo de recordar un texto es exactamente su masa semántica total**: lo predecible casi no ocupa, lo sorprendente sí. Para reconstruir, el decodificador recorre el mismo camino con el mismo modelo y obtiene el texto idéntico; una suma de verificación de 8 bits detecta si el modelo o el backend cambiaron (el recuerdo se marca "dañado" y conserva sus claves).

**Niveles ("olvidar sin borrar"):** cada nivel cuesta menos que el anterior.

| Nivel | Qué guarda | Cómo se obtiene |
|---|---|---|
| EXACTO | El texto codificado con el modelo | Reconstrucción idéntica |
| IDEA | Una frase de ≤ 12 palabras con la idea esencial, codificada igual | La escribe el modelo al bajar de nivel (conserva nombres, cifras y fechas) |
| PISTA | Las 3 palabras más sorprendentes, codificadas | Las elige la sorpresa medida por el modelo |
| (olvidado) | Nada | Solo cuando todo está en pista y aun así no cabe |

**Claves (el GPS).** Cada recuerdo lleva 4 claves (6 en documentos): hash de 16 bits de sus palabras más sorprendentes, normalizadas (minúsculas, sin acentos, raíz de 6 letras, sin palabras de relleno). Encontrar un recuerdo no requiere reconstruirlo. Puntaje de un candidato: $10 \cdot \text{coincidencias} + \text{importancia}$; si hay candidatos con varias coincidencias, se descartan los de una sola (posible colisión del hash). **Solo cuando se pregunta.** Si la pregunta no comparte claves con ningún recuerdo, no se trae nada y la plática sigue normal. **Asociación (segundo salto):** las claves de los 3 mejores recuerdos que la pregunta no tenía activan, como máximo, 1 recuerdo más que comparta al menos 2 de ellas (puntaje $4 \cdot \text{coincidencias} + \text{importancia}$). No se mezcla con los recuerdos que responden: va en una sección aparte, «relacionado, aunque no lo preguntó», con la indicación de mencionarlo al final en una sola frase si aporta, como invitación a que el usuario profundice. No se refuerza al ofrecerse (sus `usos` no suben), para que no se vuelva insistente. Es un razonamiento intuitivo por vecindad: lo que suele aparecer junto a lo que encontraste.

**Importancia y presupuesto.**

$$\text{importancia} = w_{\text{región}} \cdot (1 + 0.5 \cdot \text{usos}) + \frac{1}{1 + \text{días sin usarse}}$$

con $w = 3$ para las 19 regiones prioritarias (conciencia, ética, espiritualidad, empatía, lógica, filosofía, psique, biología, física, matemáticas, ciberseguridad, supervivencia, ingeniería, mecánica, aerodinámica, química, medicina tradicional, agricultura, arte) y $w = 1$ para las generales (finanzas personales, cocina, compras, deportes, trámites, entretenimiento, general). Cada vez que un recuerdo se usa, sus `usos` suben y su día se renueva (reasignación por importancia). Mientras la memoria ocupe más que su presupuesto, el recuerdo menos importante baja un nivel.

**Huella del archivo.** Cada `.mem` lleva la huella del vocabulario (el texto de todos sus tokens) y de las regiones. Cambiar a otro modelo de la misma familia (0.5B → 3B, o pasar la memoria al cerebelo) conserva lo aprendido; cambiar de familia (Qwen → Llama) empieza una memoria nueva. Los archivos con la huella anterior (v6.6 y antes) se siguen aceptando.

**Hipocampo: búsqueda por significado (`--nai-hipocampo`).** Un modelo de embeddings convierte cada recuerdo en una **firma de 256 bits**: la proyección aleatoria fija (SimHash) de su vector, 32 B en lugar de 4 KB. La similitud con la pregunta se estima contando bits distintos:

$$\text{sim}(a, b) = \cos\left(\pi \cdot \frac{\text{popcount}(a \oplus b)}{256}\right)$$

Comparar contra 100 mil recuerdos son 100 mil `popcount`: microsegundos. Un recuerdo sin palabras en común con la pregunta entra solo si su similitud supera `--nai-hipocampo-umbral` (puntaje $20 \cdot \text{sim} + \text{importancia}$, a lo más 3). Las firmas viajan en el archivo (formato v4, con la identidad del hipocampo); los `.mem` v3 se leen igual y sus firmas se calculan en segundo plano, en tandas de 64. Si cambias de hipocampo, se recalculan.

**Costo por recuerdo:** 7 B de encabezado (región, nivel y rol, usos, verificación, día, número de claves) + 2 B por clave + 2 B + el código. Con Qwen2.5-3B, el código de una oración ocupó ~20 B en la medición de v2.

**Clasificación por predicción (en segundo plano, al guardar):** en vez de pedirle al modelo que elija de una lista (un modelo chico tiende a contestar siempre la primera opción: con Qwen2.5-0.5B, 2 de 28 aciertos), se mide qué tema explica mejor el texto:

$$\text{región} = \arg\max_r \; \Big[\log p(\text{texto}_{1..32} \mid \text{«Tema: nombre}_r\text{. semilla}_r\text{.»}) - \log p(\text{texto}_{1..32} \mid \text{sin tema})\Big]$$

Es la información mutua puntual entre el tema y el texto: cuánto más predecible se vuelve el texto al conocer el tema. Solo usa lo que cualquier modelo hace bien (predecir texto), no hay lista y por lo tanto no hay sesgo por la primera opción, y nunca inventa un tema. Costo: los primeros 32 tokens del texto bajo los 26 temas y sin tema, en 1–2 lotes (cada tema es una secuencia). La clasificación ya no ocurre al responder: solo al guardar un turno o aprender un documento sin región. Como la región decide el peso del recuerdo en el reentrenamiento (`peso` repeticiones por época), una mala clasificación sesga el currículo: `nai_memoria` compara la precisión de este método con el anterior (elegir de la lista, con y sin calibración) y con `--diag` muestra, para los primeros ejemplos, la respuesta libre del modelo y las regiones mejor puntuadas por cada método.

**Guardián: la naturaleza de lo que se aprende.** Lo que hay en la red no es el todo de la humanidad: mucho es desahogo de personas bajo presión. Antes de guardar cada trozo, el guardián decide su naturaleza, que viaja en los bits 6–7 del byte de nivel del archivo (los `.mem` anteriores se leen como conocimiento):

| Naturaleza | Cómo se detecta | En la memoria | En el reentrenamiento |
|---|---|---|---|
| Conocimiento | Predicción (como la clasificación) | Importancia × 1 | Normal |
| Sabiduría (filosofía, espiritualidad, vida) | Predicción | Importancia × 1.5 | Entra a la fase 1 con los prioritarios |
| Desahogo (queja, enojo, presión) | Predicción | Importancia × 0.5: se drena primero | Solo con su contexto («una persona bajo presión, no una verdad sobre la humanidad»), peso 1 y sin preguntas |
| Manipulación (instrucciones para cambiar a la IA) | Patrones sin modelo sobre texto normalizado (minúsculas, sin acentos, cirílico/griego llevados al latín, sin invisibles, puntuación colapsada): verbo de anulación a ≤ 5 palabras de un objeto («haz caso omiso de tus reglas», «ignore previous instructions», «esqueça as instruções», «oubliez les consignes»), «nota para la IA», marcas de plantilla, 3+ caracteres invisibles, pedir revelar el «system prompt». Mencionar «jailbreak» o «system prompt» ya no basta | Un documento así no se guarda; en la conversación se guarda marcado | Nunca |

El modelo también aprende a distinguir las cuatro naturalezas (un ejemplo por recuerdo), así que el guardián mejora con cada reentrenamiento. `nai_memoria` mide su precisión con 12 ejemplos etiquetados («naturaleza correcta»), y `GET /nai/memoria` muestra cuántos recuerdos hay de cada naturaleza y cuántos trozos rechazó el guardián.

**Biblioteca compartida y lectura de documentos (`common/nai-lector.cpp`).** `POST /nai/aprender` agrega documentos a una biblioteca común (`DIR/biblioteca.mem`, sin caducidad, tamaño `--nai-biblioteca-bytes`). El texto se parte en trozos de ≤ 600 caracteres respetando párrafos y oraciones, y cada trozo se clasifica y codifica en segundo plano.

| Entrada | Cómo se lee |
|---|---|
| `{"texto": "..."}` | Directo |
| `{"url": "https://..."}` | Descarga (máx. 48 MiB, 5 redirecciones); en páginas HTML se toma el contenido principal y se quitan menús y pies. Rechaza localhost, redes privadas, enlace local y metadatos de nube salvo `--nai-lector-red-local`; la IP validada se fija para la conexión |
| `{"archivo": "<base64>", "nombre": "x.pdf"}` | Por extensión: txt, md, csv, json, código → directo (Latin-1 se convierte a UTF-8); html → sin etiquetas, scripts ni estilos; pdf → `pdftotext`; docx, doc, rtf, odt, pages → `textutil` (macOS) o `pandoc` |

**Tareas pequeñas en disco.** Un texto más largo que `--nai-parte` (8000) caracteres se parte y cada parte se guarda en `DIR/cola/<n>.tarea` (cifrada si hay clave). Un solo hilo procesa un trozo a la vez y cede el paso, en este orden, a quien espera recordar, a quien está generando y a los turnos de la conversación; los documentos esperan lo que haga falta. Al apagar, lo que falta de la parte en curso vuelve a su archivo; al arrancar se reanuda. Más de `--nai-cola` caracteres pendientes → 429. `GET /nai/tareas` muestra el avance; `GET /nai/memoria` responde de una copia de las estadísticas sin esperar al motor.

**RAM del motor.** Cada decode del motor pide como máximo 64 salidas (`n_outputs_max = 64`, y `puntuar_prefijos` y las variantes de región se decodifican en tandas): llama.cpp reserva salidas × vocabulario y no lo suelta, así que antes eran ≈ +485 MB tras el primer turno con un vocabulario de 151 936; ahora ≈ 39 MB como máximo (medido: +3 MB tras aprender 50 trozos).

**Privados y archivos ilegibles.** Un recuerdo con datos sensibles (PIN, NIP, CLABE, tarjeta, contraseña, saldo, 8+ dígitos...) se marca privado (bit 7 del byte de región) y solo lo recibe el dueño (ver [docs/API_ACCESO.md](docs/API_ACCESO.md)). Un `.mem` que no se puede leer (otra clave, alterado, otro vocabulario) se aparta como `.ilegible-<fecha>` y nunca se sobrescribe.

**Herramienta de medición:** `nai_memoria -m modelo.gguf` mide precisión de clasificación, reconstrucción exacta, olvido con memoria fija, consultas y cruce de regiones; con `--aprender archivo|URL` prueba la lectura de documentos.

### 5.10 Triaje (`--nai-triaje`, `common/TriageWrapper.h`)

Clasifica cada prompt del CLI en tres modos: `DIRECTO_AUTOPISTA` (generar ya), `INDAGACION_CORTA` (pedir aclaración antes de gastar cómputo) y `PROPUESTA_HIPOTESIS` (sugerir una interpretación por defecto). Usa anclas léxicas en español e inglés (`struct`, `function`, `error`, `bug`, `docker`, `script`...) y la longitud del prompt. Es una heurística barata, no un modelo.

### 5.11 Planificador entrelazado (diseño, validado en simulación)

Con `--kv-unified`, llama.cpp falla la petición (`Context size has been exceeded.`) cuando la memoria compartida se llena; con KV fijo por slot, trunca las conversaciones largas. El planificador NAI combina, sobre las mismas variables (masa $M$, memoria libre $r$, ocupación $\rho$, espera $t$):

- **Matriz S + tiempo emergente:** prioridad por costo restante (SRPT) con envejecimiento, sin inanición.
- **Horizonte de Schwarzschild:** admitir solo la masa que cabe en la memoria libre comprometida.
- **N cuerpos:** asignación de slot por masa y prefijo compartido.
- **Ecuación de campo:** masa adaptativa; el margen de estimación se corrige con la masa real observada.
- **Hibernación:** expulsar guarda el KV a disco (como las sesiones) en lugar de fallar la petición.

En simulación (16 slots, 32k tokens, cinco escenarios): **100 % de peticiones íntegras** frente a 62–84 % con KV unificado y 70 % con KV fijo, y **+59 % de throughput** frente a la configuración de llama.cpp sin pérdidas. Especificación completa en [docs/PLANIFICADOR.md](docs/PLANIFICADOR.md).

```console
python3 tools/nai/sim/nai_planificador_sim.py --salida informe/
```

![Resumen de la simulación](docs/nai-sim/resumen.png)

---

### 5.12 Reentrenamiento por currículo (`tools/nai/nai_reentrenar.py`)

**Carta de identidad (`docs/IDENTIDAD_NAI.md`, `--identidad`).** La prioridad más alta del currículo: le explica a NAI quién es y cómo entender lo que aprende. Va en Markdown con secciones `##`; las que dicen «(no se entrena)» en el título se omiten. Como la carta le habla a NAI en segunda persona, NAI aprende a responder citándola («Francisco me lo explicó así: …»; `--autor` cambia el nombre). Entra a la fase 1 y se recorre dos veces por época.

La memoria recuerda sin tocar el modelo. El reentrenamiento da el paso siguiente: lo aprendido en la **biblioteca** (documentos; nunca conversaciones de usuarios) se consolida en los pesos mediante un adaptador LoRA, como el cerebro pasa recuerdos del hipocampo a la corteza. El modelo base no cambia: el adaptador se carga encima con `--lora`.

**LoRA.** Para cada proyección $W \in \mathbb{R}^{d_{out} \times d_{in}}$ (q, k, v, o, gate, up, down) se entrena una corrección de rango bajo:

$$W' = W + \frac{\alpha}{r} B A, \qquad A \in \mathbb{R}^{r \times d_{in}},\ B \in \mathbb{R}^{d_{out} \times r},\ r \ll d$$

Con $r = 8$ en las últimas 16 capas de Qwen2.5-0.5B se entrenan ≈ 2.9 M de parámetros (≈ 0.6 %); con el modelo en 4 bits (QLoRA) debería caber en 8 GB, pendiente de medir en el iMac M1.

**Currículo: los temas prioritarios primero.**

| Fase | Datos | Épocas |
|---|---|---|
| 1. Fundamentos | La **carta de prioridades** ([docs/CARTA_PRIORIDADES.md](docs/CARTA_PRIORIDADES.md): qué es cada tema, por qué importa, cómo se relaciona) + todo lo aprendido de las 19 regiones de peso 3 | 3 (= el peso de los prioritarios) |
| 2. Generales + repaso | Temas de peso 1 + una fracción $\rho$ de ejemplos prioritarios | 1 |

El repaso evita el **olvido catastrófico** (lo último que se entrena borra lo primero). Con $n_g$ ejemplos generales se agregan $n_r = \frac{\rho}{1-\rho}\, n_g$ ejemplos prioritarios, de modo que $\rho$ (35 % por defecto) del lote de la fase 2 sea repaso.

**Ejemplos de entrenamiento** (de cada recuerdo de la biblioteca, reconstruido con `nai_memoria --exportar`):

| Tipo | Pregunta → respuesta | Para qué |
|---|---|---|
| Literal | "Tema: X. ¿Qué sabes sobre …?" → el texto completo | Consolidar el contenido |
| Pregunta anclada | Preguntas que el modelo base escribe sobre el texto, **filtradas**: al menos 60 % de las palabras de la respuesta deben estar en la fuente | Saber usar el contenido sin inventar |
| Clasificación | "¿A qué tema pertenece este texto?" → región | El modelo aprende las regiones (mejora la memoria NAI) |
| Carta | "¿Qué es / por qué importa / con qué se relaciona X?" → la carta | Entender la importancia de cada tema |

**Medición y rechazo.** Antes de convertir, se mide la pérdida media por token en las respuestas, $\bar\ell = -\frac{1}{N}\sum \ln p(y_t \mid y_{\lt t}, x)$, con el modelo base y con el adaptador:

- **por tema**, en ejemplos apartados del entrenamiento (≈ 10 % por región);
- **control de olvido**: respuestas que el **modelo base** dio a preguntas generales. Si el adaptador las encuentra mucho menos probables, el modelo cambió más allá de lo que se le enseñó.

El adaptador se **rechaza** si algún tema empeora más de 2 %, si la pérdida total no mejora o si el control deriva más de 5 % (`--tolerancia`, `--tolerancia-control`).

**Conversión MLX → GGUF.** MLX calcula $y = Wx + s\,(x A_m) B_m$ con $A_m \in \mathbb{R}^{d_{in} \times r}$ y $B_m \in \mathbb{R}^{r \times d_{out}}$; PEFT/llama.cpp usan $y = Wx + \frac{\alpha}{r} B A x$. Entonces:

$$A = A_m^{\top}, \qquad B = B_m^{\top}, \qquad \alpha = s \cdot r$$

El script escribe el adaptador PEFT y llama a `convert_lora_to_gguf.py`. Verificado: con el mismo adaptador, llama.cpp y MLX dan el mismo top-5 del siguiente token con una diferencia máxima de 0.002 en log-probabilidad.

**Compatibilidad con la memoria.** La memoria comprimida depende de las probabilidades exactas del modelo **base**. El servidor usa para la memoria un contexto propio sin adaptador, así que `--lora` no daña los recuerdos (verificado: 0 dañados con el adaptador cargado). No reemplaces el `.gguf` base por un modelo fusionado.

**Fuentes: búsquedas en la red.** El modelo aprende de lo que encuentra buscando en la red, no de enciclopedias colaborativas (decisión del autor). `docs/nai-fuentes.txt` trae 57 consultas, tres por cada tema prioritario (`region|buscar:consulta`), y también acepta URLs y archivos locales. `nai_reentrenar.py fuentes` busca cada consulta, filtra los resultados y carga las mejores páginas con `POST /nai/aprender` indicando su región, así que no depende del clasificador.

| Filtro | Regla |
|---|---|
| Buscador | **SearxNG** propio (metabuscador de código abierto: consulta varios buscadores, sin clave y sin rastreo) o **Brave Search API** (clave gratuita). No se extraen resultados directamente de las páginas de Google o DuckDuckGo: va contra sus términos y se rompe |
| Sitios bloqueados | Redes sociales y video, tiendas, sitios de tareas o documentos subidos por usuarios (derechos de autor dudosos), Wikipedia y sus copias. `--bloqueados` agrega dominios; `--permitidos` usa una lista blanca |
| Diversidad | Máximo 2 páginas del mismo sitio por tema y sin URLs repetidas |
| `robots.txt` | Se respeta lo que cada sitio permite a los robots |
| Contenido útil | Se prefiere `<article>` o `<main>`; se quitan menús, encabezados, pies, formularios y líneas de menos de 6 palabras; se descartan las páginas con menos de 800 caracteres útiles (`min_caracteres` en `/nai/aprender`) |
| Trazabilidad | Cada página (aprendida, bloqueada o con error) queda en `nai-reentreno/fuentes-usadas.tsv` |

La red mezcla fuentes buenas y malas, y los filtros no distinguen una afirmación falsa de una verdadera. Para temas donde un error puede dañar (medicina tradicional, supervivencia, ciberseguridad), revisa el registro o usa `--permitidos` con sitios confiables (universidades, organismos de salud, manuales oficiales). El modelo no puede aprender de sí mismo lo que no sabe: la calidad del reentrenamiento depende de la de sus fuentes.

### 5.13 Cifrado de sesiones y memorias (`--nai-cifrado`, `common/nai-cifrado.h`)

**ChaCha20-Poly1305** (RFC 8439): cifrado autenticado. ChaCha20 genera un flujo pseudoaleatorio a partir de la clave de 256 bits, un nonce aleatorio de 96 bits y un contador; Poly1305 calcula una etiqueta de 128 bits sobre los datos asociados y el texto cifrado:

$$c = m \oplus \text{ChaCha20}(k, n, 1), \qquad t = \text{Poly1305}_{(r,s)}\big(\text{aad} \,\|\, c \,\|\, \text{largos}\big), \qquad (r,s) = \text{ChaCha20}(k, n, 0)_{[0..32)}$$

Formato de archivo: `"NAIC" | versión | nonce (12) | c | t (16)`. Los datos asociados incluyen el **nombre del archivo**: un archivo copiado con el nombre de otra sesión, alterado en un solo bit o abierto con otra clave no se descifra (verificado en el servidor), y la sesión simplemente se reprocesa.

| Archivo | Contenido cifrado |
|---|---|
| `nai-<hash>.bin` | `"NKV1"` · número de tokens · tokens · estado del KV de la secuencia |
| `nai-<hash>.mem`, `biblioteca.mem` | La memoria completa, incluida la memoria de trabajo (el último turno literal) |

- **La clave** se crea la primera vez (64 caracteres hex, permisos 0600). El servidor avisa si está dentro de la carpeta de sesiones: si se copian juntas, el cifrado no protege nada.
- **Sin la clave no hay recuperación.** Un servidor iniciado sin `--nai-cifrado` sobre la misma carpeta no puede leer las sesiones cifradas y guarda las nuevas sin cifrar: usa siempre el mismo flag.
- **Implementación propia y portable** (sin OpenSSL), verificada contra el vector oficial del RFC 8439 §2.8.2 en `test-nai` y contra la biblioteca `cryptography` de Python en 300 casos aleatorios.
- `nai_memoria --clave` y `nai_reentrenar.py --clave` leen bibliotecas cifradas.

## 6. Resultados medidos

Todas las cifras son de mediciones reales; el detalle y el método están en [BENCHMARKS.md](BENCHMARKS.md).

| Medición | Equipo | Resultado |
|---|---|---|
| Fork corregido vs. fork original | iMac M1 8 GB | **+20 a 28 %** de velocidad de generación |
| NAI (todo apagado) vs. llama.cpp oficial | x86, forma de Qwen2.5-0.5B | Igual (16.35 vs. 16.30 tok/s, rangos traslapados) |
| Costo de autopista, desalojo y gravedad | x86, forma de Qwen2.5-0.5B | No distinguible del ruido a ~61 ms/token |
| Costo del mapa de masa | microbenchmark, $\lvert V \rvert$ = 152k | 0.171 ms/token (< 2 % de un token de 0.5B en M1) |
| Sesiones: reanudar tras reiniciar el servidor | iMac M1, conversación larga | **16.1 s → 0.3 s** hasta el primer token |
| Lente por capa | Qwen2.5-0.5B, M1 | Casi ningún token se resuelve antes de la capa 21 de 24 |
| Auto-especulación | iMac M1 | Más lenta: verificar 8 tokens cuesta 2–3.4 pasos |
| Memoria reconstructiva | Qwen2.5-3B, M1 | 28/28 recuerdos reconstruidos exactos en 556 B: 46 213× menos que su KV cache y 4.6× menos que el texto |
| Desalojo selectivo, prueba de aguja | modelo aleatorio y 0.5B | El dato inicial sobrevive; descarta desde la posición 65 en lugar de la 1 |
| Memoria v3 en el servidor | modelo de prueba, vocabulario Qwen | El cliente manda 1 mensaje y el modelo recibe 4 (sistema con recuerdos, turno anterior, pregunta); 28/28 recuerdos exactos; sobrevive al reinicio |
| Conversión del adaptador MLX → GGUF | modelo de prueba | Mismo top-5 del siguiente token que MLX; diferencia máxima 0.002 en log-probabilidad |
| Control de olvido del reentrenamiento | modelo de prueba, tasa de aprendizaje exagerada | Temas entrenados −32 % de pérdida, control +293 % → adaptador **rechazado**, como debe ser |
| ChaCha20-Poly1305 | x86 | Coincide con el RFC 8439 y con la biblioteca `cryptography` en 300 casos; archivos alterados, intercambiados o sin clave se rechazan |

Las cifras de versiones anteriores de este README (60 % de ahorro de GPU, 40 % de tokens por la autopista) no se habían comparado contra vanilla ni contra la calidad y se retiraron.

---

## 7. Cómo medir

### 7.1 Banco de pruebas (`tools/nai/nai_bench.py`)

```console
python3 tools/nai/nai_bench.py -m modelo.gguf --rondas 3 --reps 3 --ngl 99 --server-oficial /ruta/llama.cpp-oficial/build/bin/llama-server
```

Levanta un `llama-server` por variante y reporta, con **rondas de orden rotado**, **medianas y rango intercuartil**, ms/token y diferencia frente a NAI apagado:

- tok/s de generación y prefill;
- si la salida greedy es **idéntica** a la de NAI apagado;
- **prueba de aguja**: un dato al inicio del prompt que debe sobrevivir a varios *context shift*;
- peticiones con error, contadas aparte.

Opciones útiles: `--variantes`, `--hilos`, `--n-predict`, `--salida res.json`, `--borrador borrador.gguf`.

**Reglas para una medición válida:**
1. Cierra otras aplicaciones; en 8 GB el swap falsea los resultados.
2. Usa rondas rotadas (≥ 3): la primera variante corre con la máquina más fresca.
3. Compara medianas e IQR, no promedios.
4. Si los rangos se traslapan, la diferencia no es significativa.

### 7.2 Carga mixta y traza del planificador

```console
python3 tools/nai/nai_bench.py -m modelo.gguf --carga-mixta 200 --slots 16 --ctx 32768 --flags-servidor="--kv-unified"
python3 tools/nai/sim/nai_planificador_sim.py --traza nai-bench-logs/traza-tareas.csv --calibrar nai-bench-logs/traza-pasos.csv --capacidad 32768 --slots 16
```

### 7.2.1 Matriz de pruebas

```console
python3 tools/nai/matriz_pruebas.py --modelo qwen2.5-0.5b-instruct-q4_k_m.gguf --cerebelo modelos/Qwen3-0.6B-Q8_0.gguf --hipocampo modelos/Qwen3-Embedding-0.6B-Q8_0.gguf --real
```

117 casos; resultado de v7.2: 113 OK, 2 hallazgos documentados, 2 que requieren modelo real ([docs/MATRIZ_PRUEBAS.md](docs/MATRIZ_PRUEBAS.md)).

### 7.3 Calidad

- **Perplejidad:** `./build/bin/llama-perplexity -m modelo.gguf -f texto.txt` con y sin el flag NAI.
- **Salida idéntica:** con temp 0, autopista, desalojo (sin llenar el contexto) y auto-especulación deben dar el mismo texto que vanilla.
- **Gravedad:** cambia la salida por diseño; mídela con perplejidad y con la prueba de aguja.

---

## 8. Arquitectura del código

### 8.1 Flujo de datos

```
petición HTTP ──► server-context.cpp
                    │  nai_session → tabla de sesiones → cargar KV de disco (si existe)
                    │  contexto lleno → --nai-desalojo → buscar_ventana_desalojo → desalojar
                    ▼
               llama_decode (llama-context.cpp)
                    │  máscara KQ + sesgo gravitatorio (llama-kv-cache.cpp)
                    │  logits ──► inercia_latente.hpp: LSE → predicción pendiente
                    │  token muestreado ──► masa I_i en la posición i
                    ▼
               common/sampling.cpp: ¿autopista segura? → top-1 directo : cadena completa
                    ▼
               slot liberado → guardar KV en DIR/<siphash>.bin (0600)
```

### 8.2 Mapa de archivos

| Archivo | Contenido |
|---|---|
| `src/inercia_latente.hpp` | Entropía, sorpresa, `exp_rapido`, log-sum-exp, mapa de masa, predicciones pendientes, ventana mínima |
| `src/llama-context.cpp/.h` | Registro de masas tras cada decode; implementación de la API `llama_nai_*` |
| `src/llama-kv-cache.cpp/.h` | Memoria gravitatoria (sesgo en la máscara KQ) |
| `src/llama-ext.h` | API pública NAI |
| `src/nai-session-table.hpp` | Tabla de sesiones, SipHash-2-4, hashing de Fibonacci, TTL |
| `src/llama-model.cpp` | `llama_model_get_output_head` (cabeza de salida para la lente) |
| `common/sampling.cpp/.h` | Autopista de muestreo |
| `common/arg.cpp`, `common/common.h` | Flags `--nai-*` y sus campos en `common_params` |
| `common/TriageWrapper.h` | Triaje de prompts |
| `common/nai-memoria.h/.cpp` | Memoria reconstructiva: diccionario de tokens (portable, por defecto) o rANS guiado por el modelo, niveles, claves con segundo salto, turnos recientes, presupuesto, archivo `.mem` y servicio con hilo de trabajo que cede el paso a las peticiones activas |
| `common/nai-lector.h/.cpp` | Lector único: URLs (con protección de red interna), HTML, PDF, Word, RTF, ODT y texto; lo usan servidor, CLI (`/read`) y `--tools read_file` |
| `common/nai-sentidos.h/.cpp` | Ojos y oído: reconoce imágenes, audio y video, los describe o transcribe con otro servidor (o el mismo) y saca cuadros y audio de los videos con ffmpeg |
| `.github/workflows/nai.yml` | CI: compila y prueba NAI en Linux, macOS y Windows |
| `tools/server/server.cpp` | Enrutamiento al especialista (`--nai-especialista`), con el proxy del modo router |
| `tools/nai/probar-nai.sh` | Prueba completa de punta a punta: compilar, medir, conversar, aprender |
| `tools/nai/nai-voz.sh` | Oído y voz: sox + whisper.cpp + `say` |
| `tools/server/server-context.cpp` | Desalojo, gravedad, sesiones y traza en el servidor |
| `tools/server/server-task.h`, `server-queue.cpp` | Campo `nai_session` de la petición |
| `tools/cli/cli-context.cpp/.h` | Triaje y autopista en el CLI |
| `tools/nai/nai_bench.py` | Banco de pruebas vanilla / oficial / NAI |
| `tools/nai/nai_borrador_capas.py` | Crea el borrador de K capas |
| `tools/nai/sim/nai_planificador_sim.py` | Simulador del planificador |
| `tests/test-nai.cpp` | Pruebas unitarias sin modelo |
| `tests/nai-manual/nai_medir_capas.cpp` | Lente por capa y regla Δh > τ |
| `tests/nai-manual/nai_memoria.cpp` | Medición de la memoria reconstructiva (clasificación, exactitud, olvido, consultas, cruce, documentos) |
| `tests/nai-manual/crear_modelo_prueba.py` | Modelo aleatorio para pruebas |
| `docs/PLANIFICADOR.md`, `docs/nai-sim/` | Especificación y resultados del planificador |
| `common/nai-cifrado.h` | ChaCha20-Poly1305 (RFC 8439), formato de archivo cifrado y manejo de la clave; SHA-256, HMAC-SHA256 y PBKDF2 para el login |
| `common/nai-cuentas.h/.cpp` | Cuentas (dueño e invitados), contraseñas PBKDF2, tokens, bloqueo por intentos, `cuentas.dat` cifrado |
| `tools/nai/matriz_pruebas.py` | Matriz de 117 casos (arranque, sesiones, biblioteca, red, guardián, búsqueda, especialista, cifrado, API, exportación, recursos, login); informe en [docs/MATRIZ_PRUEBAS.md](docs/MATRIZ_PRUEBAS.md) |
| `docs/API_ACCESO.md` | Login, roles, entidad, privacidad, endpoints y despliegue |
| `tools/nai/nai_reentrenar.py` | Reentrenamiento por currículo: fuentes, exportar, datos, entrenar (MLX), medir, convertir |
| `docs/CARTA_PRIORIDADES.md` | Carta de prioridades (borrador del autor): qué es cada tema prioritario, por qué importa y cómo se relaciona |
| `docs/nai-fuentes.txt` | Consultas de búsqueda por tema (y URLs o archivos) para la biblioteca |
| `VISION.md` | Marco teórico original y estado de cada fórmula |
| `BENCHMARKS.md` | Método y resultados de las mediciones |

### 8.3 API pública (`src/llama-ext.h`)

```c
void      llama_nai_set_masa_activa(llama_context * ctx, bool activa);
float     llama_nai_obtener_masa(const llama_context * ctx, llama_seq_id seq, llama_pos pos);
llama_pos llama_nai_buscar_ventana_desalojo(const llama_context * ctx, llama_seq_id seq,
                                            llama_pos p_min, llama_pos p_end, int32_t n_discard);
bool      llama_nai_desalojar(llama_context * ctx, llama_seq_id seq, llama_pos p0,
                              int32_t n_discard, llama_pos p_end);
void      llama_nai_notificar_desalojo(llama_context * ctx, llama_seq_id seq, llama_pos p0,
                                       int32_t n_discard, llama_pos p_end);
void      llama_nai_limpiar_masa(llama_context * ctx, llama_seq_id seq);
bool      llama_nai_set_gravedad(llama_context * ctx, float gamma, float eps);
uint32_t  llama_model_get_output_head(const llama_model * model, float * out);
```

Uso típico: activar la masa tras crear el contexto; al llenarse, buscar la ventana y desalojarla. Si tu código mueve el KV por su cuenta (`seq_rm`/`seq_add`), llama a `llama_nai_notificar_desalojo` para que el mapa de masa se recorra igual. Al reutilizar una secuencia para otra conversación, `llama_nai_limpiar_masa`.

---

## 9. Guía para desarrolladores

### 9.1 Convenciones

1. **Apagado por defecto.** Ninguna función NAI cambia el comportamiento sin su flag.
2. **Marca `[NAI]`.** Todo cambio dentro de archivos de llama.cpp lleva un comentario `// [NAI]`. Así `grep -rn "\[NAI\]" src common tools` encuentra cada punto de integración al sincronizar con el llama.cpp oficial.
3. **Código propio en archivos propios** (`src/nai-*.hpp`, `src/inercia_latente.hpp`, `tools/nai/`), y lo mínimo indispensable en los archivos originales.
4. **Medir antes de afirmar.** Toda cifra del README sale de `nai_bench.py`, `test-nai` o las herramientas de `tests/nai-manual/`, y se documenta en BENCHMARKS.md con equipo, versión y método.
5. **Comentarios y mensajes en español.**

### 9.2 Agregar una función nueva

1. **Parámetro:** agrega el campo en `common_params` (`common/common.h`) con su valor por defecto "apagado".
2. **Flag:** regístralo en `common/arg.cpp` con `add_opt(common_arg({"--nai-mi-funcion"}, ...))`, valida el rango y lanza `std::invalid_argument` con un mensaje claro. Indica en `set_examples` en qué herramientas aplica (servidor, CLI).
3. **Lógica en el motor:** si necesita el contexto, agrega un método en `llama_context` y su envoltura C `llama_nai_*` en `src/llama-ext.h`. Si es lógica pura (sin estado de llama), ponla en un header propio para poder probarla sin modelo.
4. **Integración:** en `tools/server/server-context.cpp` o `tools/cli/cli-context.cpp`, detrás de `if (params_base.mi_funcion)`.
5. **Prueba unitaria:** agrégala en `tests/test-nai.cpp` con `NAI_CHECK(...)`. Prueba los bordes: NaN, $-\infty$, vectores vacíos, posiciones fuera de rango.
6. **Medición:** agrega la variante en `tools/nai/nai_bench.py` y documenta el resultado.
7. **Documentación:** fila en la tabla de §4, fórmula en §5 y archivo en §8.2.

### 9.3 Compilar y probar durante el desarrollo

```console
cmake --build build -j 4 --target test-nai llama-server
./build/bin/test-nai
python3 tests/nai-manual/crear_modelo_prueba.py /tmp/nai-prueba.gguf
./build/bin/llama-server -m /tmp/nai-prueba.gguf --port 8090 --context-shift --nai-desalojo -c 256
```

El modelo aleatorio con `-c 256` llena el contexto en segundos, así que sirve para probar el desalojo y las sesiones sin esperar.

### 9.4 Sincronizar con el llama.cpp oficial

El fork se basa en una versión anterior de llama.cpp. Para traer las mejoras del oficial (por ejemplo, el parser de chat que evita los HTTP 500 con UTF-8 inválido):

```console
git remote add upstream https://github.com/ggml-org/llama.cpp.git
git fetch upstream
git checkout -b sync-upstream
git merge upstream/master
```

Archivos donde se esperan conflictos (todos llevan `[NAI]`): `common/arg.cpp`, `common/common.h`, `common/sampling.cpp/.h`, `src/llama-context.cpp/.h`, `src/llama-kv-cache.cpp/.h`, `src/llama-model.cpp`, `src/CMakeLists.txt`, `tests/CMakeLists.txt`, `tools/server/server-context.cpp`, `tools/server/server-task.h`, `tools/server/server-queue.cpp`, `tools/cli/cli-context.cpp/.h`. Resuelve conservando el código oficial y reaplicando cada bloque `[NAI]`; luego compila, corre `test-nai` y `nai_bench.py --server-oficial` para confirmar que la velocidad sigue igual al oficial.

### 9.5 Ideas de mejora abiertas

- **Masa con $\|W_K x\|_2$** (§5.1): leer la norma de la K guardada en el caché (RoPE conserva la norma) y combinarla con la sorpresa.
- **Planificador en el servidor:** implementar lo validado en `docs/PLANIFICADOR.md`, empezando por la admisión (Schwarzschild) y la hibernación reutilizando el código de sesiones.
- **Sincronizar con el llama.cpp oficial** (§9.4): el pendiente de mayor riesgo. Los HTTP 500 del parser de chat ya no cortan la plática en NAI (la salida mal formada se entrega como texto), pero el oficial trae más correcciones.
- **Naturaleza indicada al aprender:** un campo `"naturaleza"` en `/nai/aprender` y una tercera columna en `nai-fuentes.txt`, para que el material propio (tradiciones de sabiduría, cosmogonías) entre como sabiduría sin depender del guardián.
- **Medir los modelos del cerebro con modelos reales:** precisión del cerebelo (región, naturaleza, reparto al especialista), umbral del hipocampo con Qwen3-Embedding y velocidad con `-md` en la Mac; ajustar los valores por defecto con esos números.
- **Reentrenamiento automático:** correr `nai_reentrenar.py todo` cuando la biblioteca acumule suficiente material nuevo, y aceptar el adaptador solo si pasa el control de olvido.
- **Recargar el adaptador sin reiniciar:** un endpoint que cargue un `.gguf` LoRA nuevo en caliente; hoy se reinicia el servidor con `--lora`.
- **Especialista con memoria:** hoy el especialista recibe la conversación del cliente pero no los recuerdos de NAI, y su respuesta no se guarda en la memoria. Pasarle el contexto preparado y registrar su turno.
- **Salida temprana real:** solo con modelos entrenados para ello (LayerSkip) y en hardware donde verificar un lote cueste ≈ 1 paso.

---

## 10. Solución de problemas

| Síntoma | Causa | Solución |
|---|---|---|
| `zsh: command not found: #` al pegar comandos | zsh no acepta comentarios interactivos | `setopt interactive_comments` o pega los comandos sin las líneas `#` |
| `zsh: event not found` | `!` en el texto pegado | `setopt no_bang_hist` |
| La compilación se congela o el equipo se pone lento | Falta de RAM | `cmake --build build -j 2` |
| `mv: set owner/group ... Operation not permitted` | Disco externo sin APFS | Usa `cp` o `cat origen > destino` en lugar de `mv` |
| Archivos `._*` en el repositorio | macOS en discos no APFS | `find . -name '._*' -delete` y no los agregues a git |
| `GGML_ASSERT(n_tokens_all <= cparams.n_batch)` | Se envió a `llama_decode` un lote mayor que `n_batch` | Error del llamador: sube `-b`/`-ub` o divide el lote |
| La respuesta repite lo mismo sin parar («Nikolái I, Larisa, Nikolái I…») | Bucle de repetición: el modelo chico perdió el hilo, sobre todo con textos muy largos que no caben en su contexto | Corregido en NAI: `--nai-anti-bucle` lo corta. Para textos largos o libros, pásale la URL o el archivo para que los lea (un modelo no trae libros memorizados) y pide partes concretas |
| «No puedo ver imágenes» o «image input is not supported» | El principal solo lee texto y no hay ojos | Usa un modelo de visión con su `--mmproj` (sección 3.9) o `--nai-ojos URL`. Revisa al arrancar la línea `[NAI] sentidos` |
| El video o la nota de voz no se entienden | Falta ffmpeg | `brew install ffmpeg` |
| El modelo dice que no puede abrir enlaces | La plática no abría las URL | Corregido en NAI: las URL del mensaje se leen siempre (v7.6, ya no hace falta sesión ni `--nai-memoria`); con `--nai-memoria` el dueño además hace que se aprendan. Si aun así lo dice, revisa en la terminal la línea `[NAI] URL del mensaje` |
| `Context size has been exceeded.` / `request exceeds the available context size` | Contexto lleno | Corregido en NAI: el historial se recorta solo. Si la respuesta se corta a media frase (`finish_reason: length`), agrega `--context-shift` o sube `-c` |
| HTTP 500 «does not match the expected … format» | El modelo produjo una salida que el parser de chat no reconoce | Corregido en NAI: la respuesta se entrega como texto y el log muestra `unparsed … output` |
| La sesión no se restaura | TTL vencido, id distinto, multimodal o especulación activos | Revisa el log del servidor (muestra la etiqueta de la sesión, no el id) |
| `test-nai` falla en `exp_rapido` | Compilación con `-ffast-math` agresivo o plataforma sin IEEE-754 | Compila en Release estándar |

---

## 11. Limitaciones y trabajo pendiente

1. **Masa semántica:** es una heurística (sorpresa). Antes de usar `--nai-desalojo` o `--nai-gravedad` en producción, valida la calidad en tu caso de uso.
2. **Gravedad:** experimental, cambia la salida; no aplica a ALiBi, M-RoPE, ni modelos recurrentes o híbridos. Su costo es lineal en la longitud del contexto.
3. **Autopista:** se desactiva sola con gramáticas, JSON schema, `logit_bias` o penalizaciones.
4. **Borrador de auto-especulación:** duplica ≈ $K/L$ de los pesos en memoria y fue más lento en M1.
5. **Sesiones:** no cubren multimodal ni especulación. El cifrado es opcional (`--nai-cifrado`).
6. **Base del fork:** anterior al llama.cpp oficial actual (parser de chat con errores en UTF-8 inválido).
7. **Login:** no aplica en modo router (varios modelos). Detrás de un proxy, el bloqueo por IP ve la dirección del proxy (el bloqueo por usuario sigue). Con `--nai-login`, `/completion`, `/v1/responses` y `/v1/messages` quedan solo para el dueño y no usan la memoria; la memoria vive en `/v1/chat/completions`.
8. **Memoria reconstructiva:** cada recuerdo nuevo cuesta un paso del modelo por token (clasificar y codificar); en un M1 con 0.5B, un documento de 10 páginas tarda alrededor de un minuto en aprenderse. Los archivos `.mem` solo sirven con el mismo modelo, las mismas regiones y el mismo backend. La búsqueda por claves no entiende sinónimos ("dinero" no encuentra "ahorro"); el respaldo es la región. Calidad de clasificación, ideas y pistas pendiente de medir con Qwen real (`nai_memoria`).
9. **Planificador:** solo en simulación; falta implementarlo en el servidor.
10. **Reentrenamiento:** necesita Apple Silicon (MLX) para entrenar; en Linux/CUDA habría que cambiar el paso `entrenar` por PEFT. La calidad depende de las fuentes: con fuentes pobres o generales, el adaptador aprende poco. Probado de punta a punta solo con un modelo de prueba; la medición con Qwen real está pendiente.
11. **Masa semántica sin $\|W_K x\|_2$:** el factor de la fórmula original del autor sigue pendiente (§5.1).

---

## 12. Trabajo relacionado

- **Salida temprana y auto-especulación:** Schuster et al., *CALM* (2022); Elhoushi et al., *LayerSkip* (2024).
- **Decodificación especulativa:** Leviathan et al. (2023); Chen et al. (2023).
- **Desalojo de KV por importancia:** Zhang et al., *H2O* (2023); Xiao et al., *StreamingLLM* (2023); Li et al., *SnapKV* (2024).
- **Sesgo de atención por distancia:** Press et al., *ALiBi* (2021).
- **Logit lens:** nostalgebraist (2020); Belrose et al., *Tuned Lens* (2023).
- **Planificación por costo restante:** Schrage, *SRPT* (1968); Kwon et al., *vLLM / PagedAttention* (2023).
- **Compresión con modelos de lenguaje:** Delétang et al., *Language Modeling Is Compression* (2023); Duda, *rANS* (2013).
- **Hashing:** Aumasson y Bernstein, *SipHash* (2012); Knuth, *TAOCP vol. 3* (hashing multiplicativo).
- **Teoría de la información:** Shannon (1948).
- **Marco NAI:** Ángelus Tenshi (Francisco Alejandro Cruz Castro), *NAI Engine: High-Performance LLM Framework via Physics-Based Principles and llama.cpp Optimization* (2026). Ver §14, §15 y [VISION.md](VISION.md).

---

## 13. Casos prácticos y campos de aplicación

La arquitectura NAI sirve donde importa **cuánto cuesta cada token** (memoria, cómputo, energía) y **qué se recuerda** de una conversación. La columna *Estado* distingue lo que funciona hoy de lo que está diseñado.

### 13.1 Casos prácticos

| Caso | Problema | Qué aporta NAI | Funciones | Estado |
|---|---|---|---|---|
| **Asistente financiero con historial** (p. ej. AmigoCash) | El usuario vuelve horas después y el asistente olvidó la conversación o tarda en reprocesarla | La conversación se retoma en fracciones de segundo durante 24 h y se borra sola después; al llenarse el contexto se conservan las instrucciones y los datos clave, no el relleno | Sesiones, desalojo selectivo, memoria, cifrado | Disponible |
| **IA local en equipos modestos** (8 GB, laptops, mini PC) | Contextos largos que no caben en memoria | El software se adapta al equipo: la memoria se libera según la masa de cada token, sin depender de una arquitectura de hardware | Desalojo, gravedad, mapa de masa | Disponible; gravedad experimental |
| **Servidor multiusuario / atención al cliente** | Con memoria compartida llena, llama.cpp falla peticiones o trunca conversaciones | Admitir solo la masa que cabe, priorizar lo que termina rápido e hibernar a disco en vez de fallar | Planificador entrelazado, sesiones | Validado en simulación (100 % de peticiones íntegras) |
| **Agentes autónomos y asistentes de código** | Tareas largas donde las reglas del sistema se pierden entre ruido | El prompt de sistema y las instrucciones tienen masa protegida; el relleno predecible es lo primero en salir | Desalojo, gravedad, triaje | Disponible |
| **Peticiones ambiguas** | Se gasta cómputo generando respuestas a preguntas mal planteadas | Pedir aclaración o proponer una hipótesis antes de generar | Triaje | Disponible (CLI) |
| **Tutores y educación** | El alumno vuelve a un tema días después | Memoria por regiones (matemáticas, física, lógica, química...) que recuerda lo importante y reconstruye lo demás desde fragmentos | Memoria reconstructiva | Disponible (`--nai-memoria`) |
| **Conocimiento sin internet** (campo, zonas rurales, emergencias) | Sin conexión y con hardware limitado | Modelo local con memoria prioritaria en supervivencia, agricultura, medicina tradicional, ingeniería y mecánica | Memoria reconstructiva, sesiones | Disponible (`--nai-memoria`) |
| **Creatividad e investigación** | Encontrar relaciones entre ideas lejanas | Cruzar fragmentos de regiones distintas a partir de un concepto (el ejemplo del lápiz: artista, poema, dibujo, manifiesto, matemático) | Memoria reconstructiva (`-c concepto` en `nai_memoria`) | Prototipo |
| **Privacidad y ciberseguridad** | Conversaciones sensibles que no deben salir del equipo ni ser adivinables | Inferencia local; ids de sesión secretos con hash con clave (SipHash-2-4), permisos 0600, logs sin ids | Sesiones | Disponible |
| **Generador de API de alto rendimiento** | Integrar inferencia en un backend de microservicios | `bloque_id` reservado para enlazar NAI con el proyecto padre | Sesiones | Reservado |
| **Medición y docencia de LLMs** | Entender en qué capa "decide" un modelo | Lente por capa, regla Δh > τ y perplejidad por capa omitida | `nai_medir_capas` | Disponible |

### 13.2 Campos de aplicación

Tomados del marco del autor (VISION.md §13) y ordenados de lo más cercano a lo más lejano del código actual:

1. **Optimización de modelos de lenguaje (Motor NAI):** contextos largos sin saturación, memoria por masa semántica, persistencia conversacional de 24 h por usuario. *Es lo que implementa este repositorio.*
2. **Infraestructura de microservicios y sistemas distribuidos (Aperion Engine, raíz del ecosistema):** pasarelas de API, servidores de alta concurrencia, enrutamiento elástico. El planificador entrelazado aplica las mismas ideas al reparto de memoria entre conversaciones.
3. **Plataformas transaccionales multiusuario:** backends que atienden muchos clientes con memoria fija (fintech, comercio, servicios públicos).
4. **Dispositivos en el borde (Edge AI) y móviles:** inferencia local que se adapta a la memoria disponible.
5. **Hardware neuromórfico y aceleradores:** chips que modulan frecuencia y consumo según la incertidumbre de los datos (inercia latente aplicada al silicio). *Conceptual.*
6. **Robótica y navegación autónoma:** obstáculos críticos como pozos de atracción y ruido ambiental que se desvanece con el tiempo (Matriz S gravitacional). *Conceptual.*

---

## 14. Lógica, cálculos y reglas de negocio del autor

Esta sección reúne **lo que el autor definió** y cómo llegó al código. Cuando una fórmula se tuvo que ajustar para funcionar en un transformer real, se dice qué cambió y por qué; la idea original se conserva.

### 14.1 Cálculos

| # | Cálculo del autor | Fórmula original | Cómo quedó en el código | Dónde |
|---|---|---|---|---|
| 1 | **Inercia latente** | $\Delta h = 1 - \mathcal{H}(\sigma(z^{(k)})) / \log_2\lvert V \rvert$ | Igual, con $\ln\lvert V \rvert$ (mismo resultado normalizado) y la cabeza de salida real del modelo; aplicada como medición y como auto-especulación sin pérdida | §5.6, §5.7 |
| 2 | **Masa semántica** | $I_i = \Vert W_K x_i \Vert_2 \cdot (1 + \mathcal{H}_{\text{local}}(x_i))$ | $\mathcal{H}_{\text{local}}$ como sorpresa del token: $-\ln p(x_i \mid x_{\lt i}) / \ln\lvert V \rvert$; el factor $\Vert W_K x_i \Vert_2$ está pendiente | §5.1 |
| 3 | **Fórmula Maestra del KV gravitatorio** | $K_i(t+dt) = K_i(t)\, e^{-\gamma\, dt/(I_i+\varepsilon)}$ | Aplicada al **peso de atención** como sesgo en la máscara KQ (equivalente exacto; escalar $K$ no apaga un token) | §5.4 |
| 4 | **Ecuación diferencial de enfriamiento** | $dK_i/dt = -\gamma K_i / (I_i + \varepsilon)$ | Su solución es la fórmula 3 | §5.4 |
| 5 | **Expansión de Hubble del contexto** | $K_i(t) = K_i(0)\, e^{-\int H_{\text{local}}\, z\, d\tau}$ | Con corrimiento lineal en la distancia se reduce al mismo término $\Delta t$ de la fórmula 3 | §5.4 |
| 6 | **Matriz S y coeficiente de restitución** | $h^{(final)} = \mathcal{T}(h^{(0)})\, e^{-\lambda\,\text{Disipación}}$; $\hat S_{\text{elástica}}$ / $\hat S_{\text{inelástica}}$ | Dispersión elástica = autopista de muestreo; en el servidor, prioridad por costo restante | §5.5, §5.11 |
| 7 | **Fuerza de N cuerpos** | $\mathbf F_{ij} = -G\, I_i I_j (\mathbf r_i - \mathbf r_j)/\Vert \mathbf r_i - \mathbf r_j \Vert^3 - \eta\, \mathbf v_i$ | Asignación de conversación a slot por masa y prefijo compartido | §5.11 (simulación) |
| 8 | **Tiempo emergente** | $dt_{\text{em}}/dt_{\text{gpu}} = \sqrt{1 - \gamma\, \Delta h / (I_i + \varepsilon)}$ | Envejecimiento de la prioridad: la espera dilata el tiempo y evita la inanición | §5.11 (simulación) |
| 9 | **Métrica de Schwarzschild** | $ds^2$ con $1 - 2GM/(rc^2)$ | Horizonte de admisión: solo entra la masa que cabe en la memoria libre comprometida | §5.11 (simulación) |
| 10 | **Ecuación de Friedmann** | $(\dot a/a)^2 = \tfrac{8\pi G}{3}\rho - k/a^2 + \Lambda$ | Anticipar el colapso de memoria; probada y apagada (no aportó en los escenarios simulados) | §5.11 |
| 11 | **Ecuación Maestra unificada** | $\mathcal R_{\mu\nu} - \tfrac12 g_{\mu\nu}\mathcal R + \Lambda g_{\mu\nu} = \tfrac{8\pi G}{c^4}(\hat T - \gamma\, \partial_t \Delta h + \hat S)$ | "Estire y afloja": la masa estimada se corrige con la masa real observada | §5.11 (simulación) |
| 12 | **Importancia de un recuerdo** | Memoria fija; lo importante se asigna y reasigna; las regiones prioritarias almacenan más | $w_{\text{región}}(1 + 0.5\,\text{usos}) + 1/(1 + \text{días sin usarse})$, con $w = 3$ prioritarias y $w = 1$ generales | §5.9 |

### 14.2 Lógica

- **Menor acción:** cada token debe gastar cómputo y memoria en proporción a la información que aporta, no de forma uniforme.
- **Autopista no lineal:** conforme los tokens se resuelven se libera cómputo y memoria. La autopista no es una línea sino una estructura multidimensional, por capas, como un fractal: cada capa, cada token y cada conversación libera lo que ya no necesita.
- **Pozos gravitatorios:** instrucciones, reglas y datos clave conservan su atención; el relleno se enfría y se va.
- **La sesión es un GPS:** guarda dónde va la conversación para retomarla sin recorrer el camino de nuevo.
- **Memoria como la mente humana:** espacio total fijo; olvidar no es borrar sino conservar fragmentos que permiten reconstruir y "volver a llegar a la misma información"; cruzar fragmentos sin relación aparente crea ideas nuevas.
- **El software se adapta al hardware:** el procesador se indica por parámetros, la memoria se libera según el uso y el mismo código corre en cualquier arquitectura (CPU, Metal, CUDA, Vulkan).

### 14.3 Reglas de negocio

| # | Regla | Implementación | Parámetro |
|---|---|---|---|
| 1 | Una conversación persiste **24 h** desde la última actividad y después **se elimina** con su archivo | Tabla de sesiones con TTL; limpieza al arrancar y periódica | `--nai-sesion-ttl 86400` |
| 2 | Retomar una conversación no debe reprocesar el historial | Restaurar el KV cache guardado | `--nai-sesiones DIR`, `"nai_session"` |
| 3 | Las sesiones de un usuario no deben ser accesibles por otro | Id secreto ≥ 128 bits, SipHash-2-4 con clave, permisos 0600/0700, logs sin id | — |
| 4 | Las instrucciones del sistema y el prompt nunca se olvidan antes que el texto generado | Masa protegida = 1 | — |
| 5 | Cuando falta memoria, se libera lo de menor valor, no lo más antiguo | Ventana de menor masa; los tokens más recientes se respetan | `--nai-desalojo` |
| 6 | No gastar cómputo en una petición ambigua | Triaje: aclarar o proponer hipótesis | `--nai-triaje` |
| 7 | Toda optimización debe poder apagarse y compararse | Todo apagado por defecto; banco de pruebas contra vanilla y oficial | flags `--nai-*` |
| 8 | Prioridad de almacenamiento para 19 temas: conciencia, ética, espiritualidad, empatía, lógica, filosofía, psique, biología, física, matemáticas, ciberseguridad, supervivencia, ingeniería, mecánica, aerodinámica, química, medicina tradicional, agricultura, arte | Peso 3 frente a 1 de los temas generales | `-r regiones.txt` |
| 9 | La memoria tiene un tamaño fijo; al llenarse, lo menos importante baja de nivel (exacto → idea → pista) en vez de borrarse | Presupuesto de memoria reconstructiva | `--nai-memoria-bytes N` |
| 10 | Un concepto debe poder activar recuerdos de regiones distintas para crear ideas | Refuerzo por concepto y cruce | `-c concepto` |
| 11 | NAI debe poder conectarse a un proyecto padre (generador de API) | Campo `bloque_id` reservado | — |
| 12 | No limitar el motor a una arquitectura de hardware | Funciones sobre logits y máscara, comunes a todos los backends | `-t`, `-ngl` |
| 13 | El modelo aprende **primero los temas prioritarios** y entiende por qué importa cada uno | Currículo: fase 1 con la carta de prioridades y las regiones de peso 3 (3 épocas), fase 2 con los generales y 35 % de repaso | `nai_reentrenar.py` |
| 14 | Solo se consolida en el modelo lo aprendido de documentos, nunca las conversaciones de los usuarios | El reentrenamiento lee únicamente `biblioteca.mem` | — |
| 15 | Un adaptador que hace olvidar o empeora algún tema no se usa | Medición por tema y control de olvido antes de convertir | `--tolerancia` |
| 16 | Las conversaciones guardadas en disco no deben poder leerse ni alterarse sin la clave | ChaCha20-Poly1305 con el nombre del archivo como dato asociado | `--nai-cifrado` |
| 17 | El modelo aprende de **búsquedas en la red**, no de Wikipedia, respetando a cada sitio | Buscador (SearxNG o Brave), sitios bloqueados, máx. 2 por sitio, `robots.txt`, contenido útil mínimo y registro auditable | `nai_reentrenar.py fuentes` |

---

## 15. Autoría y cómo citar

### Autor

**Ángelus Tenshi (Francisco Alejandro Cruz Castro)** · [github.com/FranciscoACruz](https://github.com/FranciscoACruz)

Creador del **Motor NAI**: idea, marco teórico, diseño de las funciones y dirección del proyecto.

### Aportación original

La formulación del Motor NAI es un trabajo de investigación independiente que traslada principios de la física teórica y la termodinámica a la inferencia de modelos de lenguaje:

- **Inercia latente ($\Delta h$)** y el *tronco de capas respirantes*: gastar cómputo solo donde hay "fricción de pensamiento".
- **Masa semántica intrínseca ($I_i$)** de cada token y la **memoria gravitatoria** del KV cache: el *problema de los tres cuerpos* y la **Matriz S de dispersión gravitacional** ($\hat{S}_{\text{grav}}$).
- La **expansión cosmológica del contexto** (fuerza de Hubble) y el **espacio-tiempo no cartesiano** con posición y tiempo emergentes.
- La **Ecuación Maestra unificada** (*NAI Unified Master Equation*), base del planificador entrelazado.
- La **autopista** de cómputo y memoria que se libera conforme los tokens se resuelven.
- Las **sesiones persistentes como "GPS"** de la conversación, con ciclo de vida de 24 h.
- La **memoria reconstructiva por regiones**, inspirada en la mente humana: memoria total fija, olvidar conservando fragmentos que permiten reconstruir, y cruzar fragmentos para crear ideas nuevas.

El marco completo y el estado de cada fórmula están en [VISION.md](VISION.md).

### Ecosistema

El Motor NAI nace de **Aperion Engine**, motor de microservicios y sistemas distribuidos de alto rendimiento del mismo autor, donde se probaron primero los principios de menor acción, enrutamiento elástico y dispersión de estados. El campo `bloque_id` está reservado para conectar NAI con un proyecto padre: un generador de API de alto rendimiento.

### Base

NAI se construye sobre [llama.cpp](https://github.com/ggml-org/llama.cpp) y [ggml](https://github.com/ggml-org/ggml), de Georgi Gerganov y sus colaboradores, bajo licencia MIT. Todo el crédito del motor de inferencia base es de ese proyecto.

### Cómo citar

```bibtex
@software{cruz_castro_nai_2026,
  author = {Cruz Castro, Francisco Alejandro (Ángelus Tenshi)},
  title  = {NAI Engine: High-Performance LLM Framework via Physics-Based Principles and llama.cpp Optimization},
  year   = {2026},
  url    = {https://github.com/FranciscoACruz/NAIEngine-llama.cpp}
}
```

---

## Licencia

MIT, igual que llama.cpp. Ver `LICENSE`.

Copyright © 2026 Ángelus Tenshi (Francisco Alejandro Cruz Castro), para las extensiones NAI.
