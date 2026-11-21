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
- **Errores:** el oficial no tuvo ninguno, y NAI tuvo 12 de 24 en todas sus variantes. Es el parser de respuestas "Content-only" del llama.cpp en que se basa el fork, que lanza una excepción con texto que no es UTF-8 válido (el modelo aleatorio lo genera con frecuencia). El oficial reescribió `common_chat_peg_parse` y ya no falla. **Pendiente: sincronizar el fork con el llama.cpp oficial.**

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
