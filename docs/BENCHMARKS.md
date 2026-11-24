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
