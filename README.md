# Motor NAI para llama.cpp

**Inercia latente, autopista de muestreo, memoria gravitatoria del KV cache, sesiones persistentes y un planificador de memoria entrelazado (en diseño).**

NAI es una extensión de [llama.cpp](https://github.com/ggml-org/llama.cpp) creada por **Ángelus Tenshi (Francisco Alejandro Cruz Castro)**. Parte de una idea: *gastar cómputo y memoria según cuánta información aporta cada token*, inspirada en principios físicos (inercia, gravedad, disipación). El marco teórico completo está en [VISION.md](VISION.md). Este documento describe **lo que hace el código hoy**, cómo activarlo y cómo medirlo.

Todas las funciones NAI están **desactivadas por defecto**. Sin flags, el motor se comporta exactamente como llama.cpp.

---

## 1. Funciones

| Función | Flag | ¿Cambia la salida? | Qué hace |
|---|---|---|---|
| Autopista de muestreo | `--nai-autopista` | **No** con temp 0. Con temp > 0, error acotado por `1 − p_min` | Devuelve el token top‑1 sin recorrer la cadena de samplers cuando el resultado ya está decidido |
| Desalojo selectivo | `--context-shift --nai-desalojo` | Solo cuando el contexto se llena | En el *context shift*, descarta la ventana de **menor masa semántica** en lugar de la más antigua |
| Memoria gravitatoria *(experimental)* | `--nai-gravedad γ` | **Sí** | Atenúa la atención hacia tokens generados de poca masa: sesgo `−γ·Δt/(masa+ε)` |
| Auto-especulación por capas | `-md borrador.gguf` + script | **No** | Salida temprana en la capa K, verificada por el modelo completo |
| Sesiones persistentes | `--nai-sesiones DIR` | No | Guarda y restaura el KV cache de cada conversación, con caducidad (TTL) |
| Triaje (CLI) | `--nai-triaje` | Solo pide aclaración | Detecta peticiones cortas y ambiguas, y pregunta antes de generar |
| Traza del planificador | `--nai-traza PREFIJO` | No | Registra por paso y por petición las variables del planificador (KV usado, masa que entra, cola, tiempos) para calibrar el simulador |

**En diseño, validado en simulación: planificador entrelazado.** Las fórmulas de nivel servidor de VISION.md (matriz S, tiempo emergente, Schwarzschild, N cuerpos, ecuación de campo, Friedmann) trabajan juntas sobre las mismas variables para repartir la memoria KV entre conversaciones sin fallar peticiones. Especificación, resultados y plan de implementación: [docs/PLANIFICADOR.md](docs/PLANIFICADOR.md).

---

## 2. Fundamentos y fórmulas implementadas

### 2.1 Masa semántica (`src/inercia_latente.hpp`)

La masa de un token es la **sorpresa normalizada** del token que realmente quedó en esa posición:

$$I_i = \frac{-\ln p(x_i \mid x_{<i})}{\ln |V|} \in [0, 1]$$

- **Masa baja:** el token era predecible (relleno, o algo que ya está en el contexto). Es candidato a desalojo o atenuación.
- **Masa alta:** el token aportó información nueva y se conserva.
- **Protegida (1.0):** posiciones sin predicción medida, que en la práctica son el prompt y el mensaje de sistema. Nunca se desalojan ni se atenúan antes que el texto generado.

Corresponde al término de información local $\mathcal{H}_{\text{local}}$ de la masa semántica de VISION.md, medido sobre el token mismo. Los logits calculados en la posición *i* predicen el token *i+1*, así que la masa se guarda cuando ese token llega, en su posición correcta.

### 2.2 Memoria gravitatoria (`src/llama-kv-cache.cpp`)

La "Fórmula Maestra" de VISION.md:

$$K_i(t+dt) = K_i(t)\cdot\exp\!\left(-\frac{\gamma\,dt}{I_i+\varepsilon}\right)$$

se aplica sobre el **peso de atención** y no sobre el vector K. Escalar K no apaga un token: solo lleva su logit $q\cdot k$ hacia 0, y si los demás logits son negativos el token incluso gana peso. Multiplicar el peso de atención por ese factor equivale a sumar a la máscara KQ, antes del softmax:

$$b_{ij} = -\frac{\gamma\,(p_i - p_j)}{I_j + \varepsilon}$$

- Funciona con y sin flash attention, en cualquier backend.
- Los tokens protegidos no decaen: su pozo gravitatorio es indefinido.
- Con distancia lineal, el "corrimiento al rojo" de la expansión de Hubble (VISION.md §11.5) se reduce a este mismo término.
- **Experimental:** cambia la salida del modelo. Se recomiendan valores de γ entre `1e-4` y `1e-3` y medir la calidad (sección 5). No aplica a modelos con ALiBi, M‑RoPE ni memoria recurrente o híbrida.

### 2.3 Inercia latente y salida temprana

$$\Delta h^{(k)} = 1 - \frac{\mathcal{H}\big(\sigma(z^{(k)})\big)}{\ln |V|}$$

donde $z^{(k)}$ son los logits que daría el modelo si saliera en la capa *k* (cabeza de salida aplicada a la capa *k*, conocida como *logit lens*).

- **Medición:** `nai_medir_capas` evalúa la regla "salir en la primera capa con Δh > τ" para varias τ. Para cada una informa cuántos tokens saldrían antes, cuántos conservarían el token final y qué fracción de capas se ahorraría.
- **Aplicación sin pérdida:** un early exit directo cambia la salida cuando la regla se equivoca, y además las capas omitidas no dejan K/V para los tokens siguientes. NAI lo aplica como **auto-especulación**. Un borrador con las primeras K capas (exactamente la salida temprana en la capa K) propone tokens, y el modelo completo los verifica en un solo paso. El resultado es idéntico al del modelo completo.

### 2.4 Autopista de muestreo (`common/sampling.cpp`)

- Solo se activa si la cadena **no** contiene samplers que puedan cambiar el argmax: gramática, `logit_bias`, penalizaciones, DRY, XTC, typical, mirostat o dynatemp.
- **temp ≤ 0:** devuelve el top‑1, idéntico a la cadena completa.
- **temp > 0:** toma el atajo solo si $p(\text{top-1}) \ge p_{\min}$ (`--nai-autopista-p-min`, 0.99 por defecto). La distancia de variación total respecto al muestreo exacto es como máximo $1 - p_{\min}$.
- La decisión se toma en una sola pasada O(|V|), con corte temprano.

### 2.5 Planificador entrelazado (diseño)

Con `--kv-unified`, llama.cpp falla la petición (`Context size has been exceeded.`) cuando la memoria compartida se llena. Con KV fijo por slot, trunca las conversaciones largas. El planificador NAI propone:

- **Matriz S + tiempo emergente:** prioridad por costo restante, con envejecimiento.
- **Horizonte de Schwarzschild:** admitir solo la masa que cabe.
- **N cuerpos:** asignación de slot por masa y prefijo compartido.
- **Ecuación de campo:** masa adaptativa, el margen de estimación se corrige con la masa real observada.
- **Hibernación:** la expulsión guarda el KV a disco, sin perder progreso, en lugar de fallar la petición.

En simulación (16 slots, 32k tokens, cinco escenarios) logra **100 % de peticiones íntegras**, contra 62–84 % con KV unificado y 70 % con KV fijo, y **+59 % de throughput** frente a la configuración de llama.cpp sin pérdidas.

```console
python3 tools/nai/sim/nai_planificador_sim.py --salida informe/   # experimentos y gráficas
```

![Resumen de la simulación](docs/nai-sim/resumen.png)

---

## 3. Compilación

```console
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server llama-cli test-nai nai_medir_capas
ctest --test-dir build -R test-nai --output-on-failure
```

`test-nai` no necesita modelo. Cubre la entropía, la sorpresa, las predicciones pendientes, el mapa de masa, la telemetría concurrente, el triaje, SipHash (con vectores oficiales) y la tabla de sesiones.

Para probar de punta a punta sin descargar nada, hay un modelo diminuto con pesos aleatorios:

```console
python3 tests/nai-manual/crear_modelo_prueba.py /tmp/nai-prueba.gguf
```

---

## 4. Uso

```console
# Autopista + desalojo selectivo + sesiones de 24 h
llama-server -m modelo.gguf --nai-autopista --context-shift --nai-desalojo --nai-sesiones ./sesiones

# Memoria gravitatoria (experimental)
llama-server -m modelo.gguf --nai-gravedad 5e-4

# Traza real para calibrar el simulador del planificador (carga mixta concurrente)
python3 tools/nai/nai_bench.py -m modelo.gguf --carga-mixta 200 --slots 16 --ctx 32768 --flags-servidor="--kv-unified"
python3 tools/nai/sim/nai_planificador_sim.py --traza nai-bench-logs/traza-tareas.csv --calibrar nai-bench-logs/traza-pasos.csv --capacidad 32768 --slots 16

# Auto-especulación: 1) medir en qué capa se resuelven los tokens, 2) crear el borrador, 3) servir
./build/bin/nai_medir_capas -m modelo.gguf -n 128
python3 tools/nai/nai_borrador_capas.py modelo.gguf borrador.gguf --capas 12
llama-server -m modelo.gguf -md borrador.gguf --spec-draft-n-max 8 --spec-draft-n-min 1
```

### Sesiones persistentes

La petición incluye `"nai_session": "<id>"`. Cuando el slot se libera se guarda el KV cache, y se restaura cuando la sesión vuelve, mientras no pasen `--nai-sesion-ttl` segundos sin actividad (86400 = 24 h por defecto).

**Seguridad:**
- **El id es un secreto:** quien lo conozca puede recuperar la conversación. Usa un valor aleatorio de al menos 128 bits (32 caracteres hexadecimales) generado por tu backend y ligado al usuario autenticado. El servidor avisa si el id tiene menos de 16 caracteres.
- **Hash con clave:** el id se convierte en nombre de archivo con **SipHash‑2‑4** y una clave aleatoria guardada en `DIR/.nai-clave` (permisos 0600). Así nadie puede fabricar un id que choque con el de otra sesión.
- **Permisos:** la carpeta queda con permisos 0700 y los archivos con 0600.
- **Logs:** nunca registran el id, solo una etiqueta derivada del hash.
- **Sin cifrado:** los archivos no están cifrados. Si contienen datos sensibles, usa un disco o volumen cifrado.
- **Limitación:** no se restauran sesiones con multimodal ni con decodificación especulativa.

---

## 5. Cómo medir (vanilla vs. NAI)

```console
python3 tools/nai/nai_bench.py -m modelo.gguf --borrador borrador.gguf --reps 3 --n-predict 128 --salida res.json
```

Para cada variante levanta `llama-server` en el mismo equipo y reporta:

- tok/s de generación y de prefill;
- si la salida greedy es **idéntica** a vanilla;
- una **prueba de aguja**: un dato al inicio del prompt que debe sobrevivir a varios *context shift*.

Al final imprime la tabla en Markdown, lista para pegarla aquí.

**Resultados:** *pendientes de medir con un modelo real.* Las cifras de versiones anteriores de este README (60 % de ahorro de GPU, 40 % de tokens por la autopista) no se compararon contra vanilla ni contra la calidad, así que se retiraron. Una prueba de humo con el modelo aleatorio de 8 capas confirmó que la autopista y la auto-especulación producen salida idéntica a vanilla (4/4 prompts), y que el desalojo selectivo conserva el inicio del prompt (descartó desde la posición 65 en lugar de la 1). Las velocidades de un modelo de 64 dimensiones no son representativas.

---

## 6. Advertencias

1. **Gramáticas:** la autopista se desactiva sola con gramática o JSON schema.
2. **`GGML_ASSERT(n_tokens_all <= cparams.n_batch)`:** este error significa que se envió un batch mayor que `n_batch` a `llama_decode`. Es un error de uso del llamador, no una protección de memoria. Sube `-b`/`-ub` o divide el batch.
3. **Memoria del borrador:** duplica aproximadamente K/N de los pesos del modelo.
4. **Calidad del borrador:** sin entrenamiento tipo LayerSkip, la aceptación con K pequeña puede ser baja. Mide antes con `nai_medir_capas`.
5. **Masa semántica:** es una heurística. Antes de usar `--nai-desalojo` o `--nai-gravedad` en producción, valida la calidad con tu propio caso de uso (prueba de aguja, perplejidad).
6. **Costo del mapa de masa:** calcular la sorpresa de cada token generado cuesta ~0.09 ms por token con un vocabulario de 32k (log-sum-exp con corte a 16 nats, error < 0.002 en la masa). Escala con el vocabulario: ~0.4 ms con 152k, menos del 2 % en un modelo de 7B. La máscara de gravedad agrega un costo lineal en la longitud del contexto.

---

## 7. Trabajo relacionado

- **Salida temprana y auto-especulación:** Schuster et al., *CALM* (2022); Elhoushi et al., *LayerSkip* (2024).
- **Desalojo de KV por importancia:** Zhang et al., *H2O* (2023); Xiao et al., *StreamingLLM* (2023); Li et al., *SnapKV* (2024).
- **Sesgo de atención por distancia:** Press et al., *ALiBi* (2021).
- **Planificación por costo restante:** Schrage, *SRPT* (1968); Kwon et al., *vLLM / PagedAttention* (2023), con expulsión y recomputación o swap de KV.
- **Decodificación especulativa:** Leviathan et al. (2023); Chen et al. (2023).
- **Teoría de la información:** Shannon (1948).

---

## 8. Mapa del código

| Archivo | Contenido |
|---|---|
| `src/inercia_latente.hpp` | Entropía, sorpresa, mapa de masa y predicciones pendientes |
| `src/llama-context.cpp` | Registro de masas tras cada decode; API `llama_nai_*` |
| `src/llama-kv-cache.cpp` | Memoria gravitatoria (sesgo en la máscara KQ) |
| `src/llama-ext.h` | API pública NAI |
| `common/sampling.cpp` | Autopista de muestreo y telemetría |
| `common/TriageWrapper.h` | Triaje de prompts (CLI) |
| `src/nai-session-table.hpp` | Tabla de sesiones y SipHash‑2‑4 |
| `tools/server/server-context.cpp` | Desalojo selectivo, gravedad, sesiones y traza en el servidor |
| `tools/nai/nai_borrador_capas.py` | Crea el borrador de K capas |
| `tools/nai/nai_bench.py` | Banco de pruebas vanilla vs. NAI |
| `tests/test-nai.cpp` | Pruebas unitarias sin modelo |
| `tests/nai-manual/nai_medir_capas.cpp` | Lente por capa y regla Δh > τ |
| `tests/nai-manual/crear_modelo_prueba.py` | Modelo aleatorio diminuto para pruebas |
| `tools/nai/sim/nai_planificador_sim.py` | Simulador del planificador entrelazado (experimentos, barridos, calibración con trazas) |
| `docs/PLANIFICADOR.md` | Especificación del planificador: variables, ecuaciones, resultados y plan |
| `docs/nai-sim/` | Resultados de la simulación (CSV y gráficas) |

---

## Licencia

MIT, igual que llama.cpp. Ver `LICENSE`.
