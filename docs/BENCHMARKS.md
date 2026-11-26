# Benchmarks NAI

## Método

`tools/nai/nai_bench.py` levanta un `llama-server` por variante y le manda los mismos prompts (greedy, temperatura 0). Para que el ruido de la máquina no se confunda con el costo de una función:

- **Rondas con orden rotado:** ninguna variante corre siempre primero, con la máquina más fresca.
- **Medianas y rango intercuartil** en lugar de promedios.
- **Costo en ms/token** frente a NAI con todo apagado. Es comparable entre máquinas.
- **Salida greedy idéntica:** se compara el texto de cada variante con el de NAI apagado.
- **Errores:** una petición con error (p. ej. HTTP 500) se cuenta aparte y no entra en las medianas.

Con `--server-oficial` se agrega como variante el `llama-server` del llama.cpp oficial.

```console
python3 tools/nai/nai_bench.py -m modelo.gguf --rondas 3 --reps 3 --ngl 99 \
        [--server-oficial /ruta/llama.cpp-oficial/build/bin/llama-server] [--hilos 4]
```

## Resultado 5: NAI sincronizado con el oficial (2026-11-25)

NAI reaplicado sobre el llama.cpp oficial `9c003a9` (antes, base `fc6545d` de agosto). Mismo equipo y mismo modelo que el Resultado 4 (CPU de 2 núcleos, pesos aleatorios con la forma de Qwen2.5-0.5B).

### Motor (`llama-bench`, tokens por segundo)

| Prueba | Oficial | NAI antes (v7.9) | NAI sincronizado |
|---|---|---|---|
| Prefill, 512 tokens | 223.6 | 200–213 | 233.1 |
| Generación, contexto vacío | 18.6 | 16.3–18.0 | 19.0 |
| Generación con 1 024 tokens de contexto | 17.3 | 11.7 | 17.6 |
| Generación con 2 048 tokens de contexto | 16.1 | 9.5 | **16.1** (+70 %) |

### Servidor (`nai_bench.py`, 24 mediciones por variante, 64 tokens)

| Variante | Generación mediana (tok/s) | Rango intercuartil | Salida igual al oficial | Errores |
|---|---|---|---|---|
| llama.cpp oficial | 18.30 | 18.1–18.5 | — | 0 |
| NAI, todo apagado | 18.36 | 18.0–18.6 | 4/4 | 0 |
| NAI `--nai-autopista` | 18.26 | 18.0–18.5 | 4/4 | 0 |
| NAI `--context-shift --nai-desalojo` | 17.73 | 17.6–17.9 | 4/4 | 0 |

El desalojo selectivo ahora se distingue del ruido: cuesta un 3 % (calcula la masa de cada token sobre todo el vocabulario). Con la base vieja quedaba escondido detrás de su lentitud.

### Conversación larga (30 turnos, contexto de 2 048)

| Motor | Turnos | Latencia mediana 1–10 / 11–20 / 21–30 | Generación mediana |
|---|---|---|---|
| Oficial | 17 de 30 (rechaza desde el 18) | 4.5 s / 4.6 s / — | 16.3 tok/s |
| NAI antes | 30 de 30 | 5.3 s / 8.7 s / 9.9 s | 11.7 tok/s |
| **NAI sincronizado** | **30 de 30** | **4.2 s / 4.6 s / 7.9 s** | **16.8 tok/s** |
| NAI antes, con memoria | 30 de 30 | 8.3 s / 8.6 s / 9.2 s | 12.3 tok/s |
| **NAI sincronizado, con memoria** | **30 de 30** | **7.4 s / 7.6 s / 8.1 s** | **14.9 tok/s** |

En los turnos 21–30 sin memoria, NAI recorta el historial en cada turno y vuelve a leer ~1 000 tokens; con memoria eso no pasa.

### Reiniciar a media conversación (2 830 tokens)

| Motor | Tiempo hasta responder |
|---|---|
| Oficial, sin guardar | 16.1 s (reprocesa todo) |
| Oficial, guardado manual por slot | 0.8 s |
| NAI antes, automático | 2.4 s |
| **NAI sincronizado, automático** | **0.9 s** |

### Capacidades tras sincronizar

NAI tiene ahora **todas** las opciones del oficial (cero opciones del oficial faltan en NAI) más las 33 propias `--nai-*`. Lo que en el Resultado 4 era solo del oficial ya está en NAI: voz generada en el servidor (`--tts-voice`), modo agente con herramientas en contenedor (`--agent`, `--tools-runtime`, `--ui-agent`), túnel P2P (`--connect`), RPC entre equipos (`--rpc`), caché de expertos MoE, modelos en varios archivos (`--shard`) y `--reasoning-effort`.

### Funciones verificadas tras sincronizar

**Matriz de pruebas completa (`tools/nai/matriz_pruebas.py`, 117 casos):** 107 OK, 2 requieren modelo real, 1 omitido, 3 hallazgos y 4 fallas. Ninguna es nueva:
- **B15 y B16** (hallazgos ya documentados): `/completion`, Responses y Anthropic no pasan por la memoria.
- **K04** (hallazgo): apagar con documentos en cola tarda ~15 s en esta CPU porque termina la parte que está aprendiendo; nada se pierde (40/40 tras reiniciar). La versión anterior da exactamente lo mismo.
- **B03**: falla igual que antes de sincronizar (Resultado 3).
- **B12, B13 y E12**: fallan solo en la corrida completa, por la cola que deja B11 (30 turnos en 45 s) en una CPU de 2 núcleos. E12 corrido aparte da OK en las dos versiones, igual que B12 y B13 en el Resultado 3.


`test-nai` completo; sentidos de punta a punta (imagen, audio WAV y WEBM, video, URL de imagen y video, aprender archivo y URL, caché, peticiones internas); URL en la plática; memoria por sesión y recorte del historial; login con dueño (configurar, login, 401 sin token, memoria por cuenta); compilación de todos los objetivos.

## Resultado 4: llama.cpp oficial vs NAI, ambos en su última versión (2026-11-25)

- **Versiones:** llama.cpp oficial `9c003a9` (2026-11-25, rama master) y NAI `121ff69` (v7.9).
- **Equipo:** x86_64, 2 núcleos de CPU, 7 GB de RAM, sin GPU. **No es la Mac:** en Apple Silicon con Metal los números absolutos cambian; las proporciones sirven de guía, no de promesa.
- **Modelo:** pesos aleatorios con la forma de Qwen2.5-0.5B (630 M de parámetros, F16, 1.17 GiB). Mide velocidad y comportamiento, **no calidad de las respuestas**.
- **Herramientas:** `llama-bench` (motor puro), `tools/nai/nai_bench.py` (servidor, 3 rondas con orden rotado), `tools/nai/nai_conversacion.py` (30 turnos, `-c 2048`, el cliente reenvía todo el historial, como la interfaz web) y un script de reinicio (conversación de 2 830 tokens, reiniciar el servidor y continuarla).

### Motor (`llama-bench`, 2 hilos, tokens por segundo)

| Prueba | Oficial | NAI | Diferencia |
|---|---|---|---|
| Prefill, 512 tokens (2 corridas, orden invertido) | 212.5 / 221.5 | 200.3 / 212.9 | NAI −5 %, dentro del ruido de la segunda corrida |
| Prefill, 2 048 tokens | 177.6 | 180.0 | empate |
| Generación con contexto vacío | 16.6–18.4 | 16.3–18.0 | empate (−2 %) |
| Generación con 1 024 tokens de contexto | 16.8 | 11.7 | **NAI −30 %** |
| Generación con 2 048 tokens de contexto | 15.7 | 9.5 | **NAI −40 %** |
| Igual, 2 048, *flash attention* apagado | 16.4 | 12.9 | NAI −22 % |

**Causa de la caída con contexto:** las funciones NAI no la provocan. NAI no modifica `ggml-cpu` (cero líneas `[NAI]`), y la gravedad, único cambio en el camino de la atención, viene apagada por defecto. La diferencia está en la base del fork (agosto de 2026): desde entonces, el oficial cambió más de 1 800 líneas de los núcleos de CPU de `ggml` (`ops.cpp`), incluida la atención. Se corrige **sincronizando con el oficial** (§9.4 del README). Es el pendiente de rendimiento más importante.

### Servidor (`nai_bench.py`, 24 mediciones por variante, 64 tokens, prompts cortos)

| Variante | Generación mediana (tok/s) | Rango intercuartil | Prefill mediana | Salida igual al oficial | Errores |
|---|---|---|---|---|---|
| llama.cpp oficial | 17.27 | 16.9–17.5 | 100.0 | — | 0 |
| NAI, todo apagado | 17.12 | 16.8–17.7 | 92.4 | 4/4 | 0 |
| NAI `--nai-autopista` | 17.54 | 17.1–17.8 | 96.1 | 4/4 | 0 |
| NAI `--context-shift --nai-desalojo` | 17.34 | 16.8–17.7 | 96.7 | 4/4 | 0 |

Con prompts cortos, los rangos se traslapan: **empate**. Las funciones NAI no cuestan velocidad, y con temperatura 0 dan exactamente la misma salida que el oficial.

### Conversación larga (30 turnos, contexto de 2 048 tokens)

| Motor | Turnos respondidos | Qué pasa al llenarse el contexto | Latencia mediana, turnos 1–10 / 11–20 / 21–30 |
|---|---|---|---|
| Oficial | 17 de 30 | Desde el turno 18 rechaza **todas** las peticiones (HTTP 400, «exceeds the available context size») | 4.5 s / 4.6 s* / — |
| Oficial con `--context-shift` | 17 de 30 | Igual: el *context shift* no evita el rechazo de una petición que ya no cabe | 4.4 s / 4.7 s* / — |
| NAI, sin flags | **30 de 30** | Recorta el historial y sigue | 5.3 s / 8.7 s / 9.9 s |
| NAI con memoria (`--nai-memoria`, sesión por defecto, desalojo) | **30 de 30** | No crece: envía recuerdos y turnos recientes (~410 tokens por turno) | 8.3 s / 8.6 s / 9.2 s |

\* Solo los turnos que respondió.

En los turnos que ambos responden, NAI tarda más por la generación más lenta con contexto largo (tabla del motor). Con memoria, la latencia queda **plana** de principio a fin, porque lo que llega al modelo no crece.

### Reiniciar el servidor a media conversación (2 830 tokens)

| Motor | Cómo se recupera | Tokens a reprocesar al continuar | Tiempo hasta responder |
|---|---|---|---|
| Oficial | No se recupera: reprocesa todo | 2 848 | 16.1 s |
| Oficial con `--slot-save-path` | El cliente debe llamar `/slots/0?action=save` antes de apagar y `restore` al volver (un archivo por slot) | 18 | 0.9 s |
| NAI `--nai-sesiones` | Automático por id de sesión, con caducidad (TTL) y cifrado opcional; el cliente no hace nada | 18 | 2.4 s |

NAI recupera solo, pero restaurar le toma ~1.1 s más que la restauración manual del oficial. Queda por revisar (lectura y validación del archivo de sesión).

### Arranque y memoria RAM

| | Oficial | NAI |
|---|---|---|
| Arranque hasta `/health` = ok | 1.22 s | 1.41 s |
| RAM con el modelo cargado | 2 010 MiB | 2 006 MiB |
| RAM tras una petición de 2 830 tokens | 2 046 MiB | 2 042 MiB |

### Capacidades

| Capacidad | Oficial | NAI |
|---|---|---|
| Motor, cuantizaciones, backends (CPU, Metal, CUDA, Vulkan…) | Sí | Sí (misma base, más antigua) |
| API compatible con OpenAI, interfaz web, *router* de varios modelos | Sí | Sí |
| Imágenes, audio y video con un modelo que los entiende (`--mmproj`) | Sí; video con opciones de ffmpeg (`--video-fps`) | Sí |
| Ver y oír con un principal que solo lee texto (ojos y oído aparte) | No | Sí (`--nai-ojos`, `--nai-oido`) |
| Recordar lo que vio y oyó, y aprenderlo en la biblioteca | No | Sí |
| Leer las URL que se pegan en el chat (HTML, PDF, imagen, audio, video) | No por sí solo (descarga las imágenes de `image_url`); en modo agente, con un servidor MCP de navegación que tú agregues | Sí, sin configurar nada, con protección de red interna, y las aprende |
| Conversación sin límite por contexto | No (rechaza al llenarse) | Sí (recorte + memoria) |
| Memoria por sesión que sobrevive reinicios, biblioteca de documentos | No | Sí (`--nai-memoria`, `/nai/aprender`) |
| Guardar y restaurar el estado | Manual por slot (`--slot-save-path`) | Automático por sesión, con TTL y cifrado |
| Guardián contra manipulación al aprender, anti-bucle | No | Sí |
| Cuentas (dueño e invitados), cifrado de memorias | Solo `--api-key` | Sí (`--nai-login`, `--nai-cifrado`) |
| Cerebelo, hipocampo, especialista en código | No | Sí |
| Arranque de todo con un comando, descarga de modelos, reinicio automático | No | Sí (`./nai.sh`) |
| Generación de voz en el servidor (TTS, `--tts-voice`) | Sí | No (NAI usa `say` por fuera) |
| Modo agente con herramientas y MCP (`--agent`, `--tools-runtime` en contenedor, `--ui-agent`) | Sí, más completo | Parcial (`--tools`, `--agent` de la base) |
| Túnel P2P (`--connect`), RPC entre equipos (`--rpc`), caché de expertos MoE, modelos en varios archivos (`--shard`), `--reasoning-effort` | Sí | No |
| Velocidad de generación con contexto largo en CPU | Referencia | 22–40 % menor (base antigua de ggml); **igual tras sincronizar (Resultado 5)** |

**En resumen:** NAI iguala al oficial en velocidad pura con contexto corto y en uso de RAM, y es el único de los dos que no se cae en conversaciones largas, recuerda entre reinicios, aprende y ve y oye con modelos chicos. El oficial es más rápido generando con mucho contexto y trae funciones nuevas (TTS en el servidor, agentes, RPC, MoE) que el fork aún no tiene. Las dos diferencias en contra de NAI se cierran **sincronizando con el oficial**.

## Resultado 1: modelo con la forma de Qwen2.5-0.5B, solo CPU (2026-11-21)

- **Modelo:** aleatorio con la forma exacta de Qwen2.5-0.5B (vocabulario de 151 936, 24 capas, 896 dimensiones, 14/2 cabezas, FFN 4864, F16), creado con `tests/nai-manual/crear_modelo_prueba.py`.
- **Máquina:** x86_64 con 2 hilos.
- **Versiones:** llama.cpp oficial `e6d624b` (2026-11-22) y NAI `064abfd` + log-sum-exp vectorizado.
- **Medición:** 3 rondas × 2 repeticiones × 4 prompts = 24 mediciones por variante, 64 tokens cada una.

| Variante | Generación mediana (tok/s) | Rango intercuartil | ms/token | Δ ms/token | Salida = NAI apagado | Errores |
|---|---|---|---|---|---|---|
| llama.cpp oficial | 16.30 | 14.9–16.9 | 61.37 | +0.21 | 2/2 | 0 |
| NAI, todo apagado | 16.35 | 15.8–17.3 | 61.16 | — | — | 12 |
| `--nai-autopista` | 16.31 | 16.1–16.6 | 61.32 | +0.16 | 2/2 | 12 |
| `--nai-desalojo` | 16.44 | 16.2–16.7 | 60.81 | −0.35 | 2/2 | 12 |
| `--nai-gravedad 5e-4` | 16.65 | 16.0–16.8 | 60.06 | −1.10 | 0/1 (cambia la salida, esperado) | 15 |

**Lectura:**
- Los rangos intercuartiles se traslapan: **ninguna función NAI tiene un costo distinguible del ruido** a ~61 ms por token.
- NAI con todo apagado **iguala al llama.cpp oficial más reciente** en generación.
- **Errores (corregido en v4):** el oficial no tuvo ninguno y NAI 12 de 24. Era el parser "Content-only" de la base del fork, que lanzaba HTTP 500 con texto que no es UTF-8 válido. El arreglo en `common/chat.cpp` devuelve el texto tal cual cuando no hay parser. Verificación posterior: **0 errores** en todas las variantes, y NAI apagado da **la misma salida que el oficial** (4/4).

## Resultado 3: conversación larga y matriz de pruebas, v7.2 vs v7.3 (2026-11-24)

Equipo: 2 núcleos de CPU, sin GPU; modelo con la forma de Qwen2.5-0.5B (pesos aleatorios: mide velocidad y comportamiento, no calidad); cerebelo del mismo modelo e hipocampo de prueba. v7.3 = v7.2 + el trabajo pesado de la memoria fuera del candado (clasificar, codificar y firmar sin bloquear las preguntas), un candado propio por modelo, el fondo cede el paso entre trozos, los turnos recientes se actualizan al instante, anti-bucle y URL en la plática.

**Matriz (`tools/nai/matriz_pruebas.py --solo B,C,H`), cada versión sola en la máquina:**

| Caso | v7.2 | v7.3 |
|---|---|---|
| B11: 30 turnos seguidos enviando solo el mensaje nuevo | FALLA: 20 de 30 respuestas en 2,349 s (cada pregunta esperaba a que el fondo guardara la anterior) | OK en 45 s |
| B19: caducidad (TTL) de una sesión | FALLA | OK |
| C08: preguntar mientras aprende un documento de 2 MB | OK; chat 24.5 s antes y 4.5 s durante | OK; chat 0.77 s antes y 1.27 s durante |
| B03: aislamiento entre sesiones | FALLA | FALLA (igual; no lo introduce v7.3) |
| B12, B13, C01, H01 | OK / FALLA / OK / OK | FALLA en la corrida completa; **OK corridos sin B11 antes** |

Por qué B12, B13, C01 y H01 fallan en la corrida completa de v7.3: B11 contesta 30 turnos en 45 s y deja unos 30 turnos por guardar (~12 s cada uno en esta máquina); esos casos esperan como máximo 300 s a que la cola se vacíe. Reproducidos por separado, el guardado rinde igual en las dos versiones (20 sesiones simultáneas: 19 y 20 archivos en ~240 s) y C y H pasan completos (0 fallas). Con atraso en la cola, los turnos recientes igual llegan a la siguiente pregunta: se actualizan al instante.

**Conversación larga (`tools/nai/nai_conversacion.py`, 20 turnos, `-c 2048`, el cliente reenvía todo el historial):** el llama.cpp oficial (`c74c71a`) responde 17 de 20 y desde el turno 18 rechaza la petición («request exceeds the available context size») para siempre; NAI responde 20 de 20 en todas las variantes. Con memoria, la latencia de los turnos 11–20 bajó de 37.1 s (antes del arreglo) a 10.3 s, y a 8.4 s con cerebelo e hipocampo. Motor puro (`llama-bench`): empate dentro del ruido (~190 tok/s de prefill, ~17 tok/s de generación).

## Costo aislado del mapa de masa (microbenchmark, |V| = 151 936)

| Versión | ms por token generado |
|---|---|
| v2 (corte a 16 nats, escalar) | 0.348 |
| **v3 (16 acumuladores + `exp_rapido` sin ramas)** | **0.171** |

El log-sum-exp v3 tarda 0.15 ms con logits con pico y 0.46 ms con logits planos (el peor caso). Su error es < 1e‑4 nats; el de `exp_rapido` es < 1e‑6 relativo en [−16, 0].
Referencia: Qwen2.5‑0.5B en un M1 genera a ~9.5 ms por token, así que el mapa de masa cuesta < 2 %.

## Resultado 2: Qwen2.5‑0.5B real en M1 8 GB (CPU, primera versión del banco)

Corrida sin rondas rotadas: variantes en bloque, promedio, con el sistema bajo presión de memoria (7 GB de swap).

| Variante | tok/s | Salida = vanilla | Aguja |
|---|---|---|---|
| vanilla | 105.7 ± 9.4 | 4/4 | sí |
| autopista | 97.0 ± 8.5 | 4/4 | — |
| desalojo | 84.8 ± 6.8 | 4/4 | sí |
| gravedad | 89.3 ± 2.9 | 0/4 | sí |

El prefill también bajó en las variantes NAI (de 237 a 161–193 tok/s), aunque la autopista no lo toca. Eso indica ruido de orden y de memoria: **hay que repetir esta medición con rondas rotadas.**

## Comparar con otros motores en Apple Silicon

Mismo modelo base (Qwen2.5‑0.5B‑Instruct a 4 bits), un motor a la vez:

```console
# MLX (motor de Apple)
pip3 install mlx-lm
mlx_lm.benchmark --model mlx-community/Qwen2.5-0.5B-Instruct-4bit -p 512 -g 128

# Ollama (usa llama.cpp por dentro)
ollama run qwen2.5:0.5b --verbose     # escribe un prompt; al final muestra "eval rate"
```

La cuantización no es idéntica (MLX 4‑bit frente a GGUF Q4_K_M), así que la comparación es orientativa.
