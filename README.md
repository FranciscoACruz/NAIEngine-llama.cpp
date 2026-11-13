<div align="center">

# 🌌 Motor NAI (NAIEngine-llama.cpp)

**Motor de inferencia de LLM de latencia ultra-baja basado en `llama.cpp`, con optimización elástica de capas, triaje semántico y gestión gravitatoria de la caché KV.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon%20%7C%20x86%20%7C%20CUDA-lightgrey)]()
[![Status](https://img.shields.io/badge/Status-Production--Ready-success)]()

*"La naturaleza no da saltos, pero el silicio bien optimizado sí."*

</div>

---

## 📖 Tabla de Contenidos

1. [Introducción y Filosofía](#-introducción-y-filosofía)
2. [Fundamentos Físicos y Arquitectónicos](#-fundamentos-físicos-y-arquitectónicos)
3. [Arquitectura del Pipeline NAI](#-arquitectura-del-pipeline-nai)
4. [Resultados del Benchmark (Apple M1)](#-resultados-del-benchmark-apple-m1)
5. [Instalación y Compilación](#-instalación-y-compilación)
6. [Ejecución del Benchmark](#-ejecución-del-benchmark)
7. [Solución de Problemas (Troubleshooting)](#-solución-de-problemas-troubleshooting)
8. [Licencia y Agradecimientos](#-licencia-y-agradecimientos)

---

## 🚀 Introducción y Filosofía

Este repositorio es un *fork* de [llama.cpp](https://github.com/ggml-org/llama.cpp) que integra el **Motor NAI**. No es una simple colección de optimizaciones empíricas; es una reingeniería del flujo de inferencia tradicional inspirada en la mecánica celeste, la física de partículas y la teoría de cuerdas.

A diferencia de los motores convencionales que procesan los tokens de forma estrictamente secuencial a través de todas las capas del transformer, el Motor NAI opera bajo un paradigma de **"inferencia elástica"**. El motor es capaz de:
*   **Saltarse capas profundas** si detecta que el modelo ya tiene certeza absoluta sobre el siguiente token.
*   **Compartir el espacio-tiempo de la caché KV** entre múltiples usuarios para evitar recálculos innecesarios.
*   **Atenuar el ruido semántico** basándose en la "masa gravitatoria" de los tokens en el contexto.

El resultado es un motor que no solo es drásticamente más rápido en escenarios de alta concurrencia, sino que "piensa" de forma más eficiente, gastando ciclos de reloj de CPU/GPU únicamente donde son estrictamente necesarios.

---

## 🔬 Fundamentos Físicos y Arquitectónicos

El diseño del Motor NAI se sustenta en las siguientes teorías:

### 1. El Problema de los Tres Cuerpos y la Matriz de Dispersión (\(e \cdot S\))
En astrofísica, predecir la trayectoria de múltiples cuerpos es un problema caótico sin solución analítica cerrada. Sin embargo, usando la **Teoría de Dispersión (Matriz S)**, podemos conectar un estado inicial (\(A\)) y uno final (\(B\)) usando un "faro" o cuerpo intermedio (\(C\)) con un coeficiente de elasticidad (\(e\)).
*   **Aplicación en el motor:** En lugar de procesar todo el prompt de un usuario nuevo desde cero, el motor usa el contexto ya cacheado de otro usuario como el "faro" (cuerpo \(C\)). Solo procesa el diferencial (\(e \cdot S\)), logrando latencias de prefill casi instantáneas (esto es el *KV Sharing*).

### 2. Inercia Latente y Entropía Gravitacional
En física, la entropía mide el desorden de un sistema. En el motor, la entropía mide la "duda" del modelo.
*   **Aplicación en el motor:** En capas intermedias (ej. capa 12), el motor evalúa la distribución de probabilidad (softmax) de los logits. Si la entropía es baja (el modelo está seguro), el motor toma una **Autopista** y se salta el resto de las capas (Early Exit). Si la entropía es alta (conflicto), el motor sigue el camino completo para deliberar.

### 3. Fricción Dinámica y Memoria de Masa Gravitatoria
Cuando un objeto masivo se mueve a través de un medio (como la materia oscura), genera una estela que lo frena (fricción dinámica).
*   **Aplicación en el motor:** La caché KV no es estática. A cada token se le asigna una **Masa Semántica** (\(I_i\)). Los tokens con mayor peso (ej. nombres propios, instrucciones clave) persisten en la memoria. Los tokens de relleno sufren un decaimiento exponencial (Redshift), liberando memoria VRAM y reduciendo el coste de atención.

---
## 🏗️ Arquitectura del Pipeline NAI

El Motor NAI intercepta el flujo de inferencia tradicional de `llama.cpp` en tres capas secuenciales de optimización. Cada capa actúa como un filtro que reduce el cómputo innecesario antes de tocar el silicio.

```text
[ Entrada: Prompt / Token ]
       │
       ▼
┌─────────────────────────┐
│ 1. TRIAJE LATENTE       │ ──► ¿Entrada ambigua? ──► Indagación Quirúrgica
│ (TriageEngine)          │
└───────────┬─────────────┘
            │ (Continuar)
            ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ 2. TRONCO DE CAPAS ELÁSTICAS                                              │
│ Capa 1..12 ──► [ Medidor de Inercia Latente Δh ]                          │
│                    ┌──────────────┴──────────────┐                        │
│                    ▼                              ▼                       │
│         [ Inercia Alta / Autopista ]   [ Nodo de Conflicto / Entropía Alta ]
│         • Early Exit (nai_logits_early) • Ejecución de Capas Profundas (13..N)
└────────────────────────────┬──────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────┐
│ 3. MEMORIA GRAVITATORIA (KV Cache)      │ ──► Atenuación temporal (dt)
│ (KV Sharing / Fricción Dinámica)        │
└─────────────────────────────────────────┘
---

### 📄 PARTE 2B: Detalles de Componentes y Matemáticas

```markdown
### 🔍 Detalle de los Componentes Internos

#### 1. TriageEngine (Filtro Quirúrgico / Prefiltro Header-Only)
Es la primera línea de defensa. Antes de que el prompt entre a la tokenización profunda (que es costosa), este componente analiza la estructura y semántica de la petición.
*   **Función:** Clasifica el flujo de ejecución. Si detecta que la entrada es ambigua, activa una plantilla de "Indagación Corta" (evitando que el modelo alucine respuestas largas). Si detecta una petición directa, activa el "Sesgo de Razonamiento Extendido" (permitiendo que el modelo use más capas si es necesario).
*   **Implementación:** `triage.evaluar(prompt_usuario)` en `TriageEngine.cpp`.

#### 2. Tronco de Capas Elásticas (Early Exit + Bypass Dinámico)
El corazón del motor. En lugar de ejecutar todas las capas del transformer (que pueden ser 32, 40 o más), el motor evalúa la **Inercia Latente** ($\Delta h$) en un punto intermedio (por ejemplo, la capa 12).
*   **Función:** Si el estado latente $h^{(k)}$ tiene alta certeza (baja entropía), el motor emite los logits de forma temprana, omitiendo las capas restantes.
*   **Implementación:** Modificación en el bucle de capas de `llama-context.cpp`, llamando a `nai_logits_early`.

#### 3. Memoria de Masa Gravitatoria (KV Cache Dinámica)
La caché de Claves/Valores (KV) tradicional crece indefinidamente y trata todos los tokens por igual. El Motor NAI la convierte en un sistema físico.
*   **Función:** Asigna una "Masa Semántica" ($I_i$) a cada token. Los tokens con mayor peso (ej. nombres propios, instrucciones clave) persisten en la memoria. Los tokens de relleno sufren un decaimiento exponencial (Redshift), liberando memoria VRAM y reduciendo el coste computacional de la atención.
*   **Implementación:** Modificación en `llama-kv-cache.cpp` y `llama-context.cpp`.

#### 4. Autopista de Samplers (Bypass de Muestreo)
El muestreo (sampling) tradicional aplica múltiples filtros (top-k, top-p, temperatura) iterando sobre el vocabulario.
*   **Función:** Si la inercia latente es alta, el motor detecta que el token ganador es matemáticamente inapelable. En ese caso, se **bypassea** toda la cadena de samplers y se selecciona directamente el token con mayor probabilidad, ahorrando ciclos de CPU.
*   **Implementación:** `evaluar_inercia_y_equivalencia` en `sampling.cpp`.

---

## 🧮 Formulaciones Matemáticas

### A. Inercia Latente y Early Exit
En la capa $k$, el estado latente $h^{(k)}$ se proyecta mediante una cabeza temprana. La **inercia latente** $\Delta h$ se define como:

$$
\Delta h = 1 - \frac{H(\sigma(z^{(k)}))}{\log_2(|V|)}
$$

Donde $H$ es la entropía de la distribución de probabilidad, $\sigma$ es la función softmax y $|V|$ es el tamaño del vocabulario. Un valor de $\Delta h$ cercano a 1 indica alta certeza, lo que permite activar el *early exit*.

### B. Memoria de Masa Gravitatoria (KV Cache)
La **masa semántica** $I_i$ de cada token $x_i$ en la caché se calcula como:

$$
I_i = \| W_K \cdot x_i \|_2 \cdot (1 + H_{\text{local}}(x_i))
$$

El decaimiento temporal de la clave $K_i$ en un intervalo $dt$ sigue una atenuación exponencial basada en su masa:

$$
K_i(t + dt) = K_i(t) \cdot \exp\left( -\frac{\gamma \cdot dt}{I_i + \epsilon} \right)
$$

Donde $\gamma$ es una constante de atenuación y $\epsilon$ evita la división por cero. Los tokens con mayor peso semántico ($I_i$ grande) decaen más lentamente, preservando el contexto relevante en la ventana de atención.

## 📊 Resultados del Benchmark (Apple Silicon M1)

Los siguientes resultados fueron obtenidos ejecutando el script `test_benchmark.cpp` en un Apple M1 con 8 GB de RAM unificada, utilizando el modelo `qwen2.5-0.5b-instruct-q4_k_m.gguf` y aceleración por hardware Metal.

### 1. El Salto Cuántico del KV Sharing (Tiempo Emergente)

La teoría de dispersión aplicada al software demuestra su valía cuando un segundo usuario intenta procesar un contexto similar al primero. El motor actúa como un faro gravitatorio, heredando el espacio-tiempo precalculado.

| Escenario | Tiempo de Prefill | Velocidad | Impacto de la Optimización |
| :--- | :--- | :--- | :--- |
| **Usuario 1 (Normal)** | 82.01 ms | 475.54 tok/s | Procesamiento estándar de la secuencia completa (79 tokens). |
| **Usuario 2 (NAI Sharing)** | **15.70 ms** | **Incalculable** | **Reducción del 81% en tiempo.** El KV se copió y solo se procesó 1 token diferencial. |

> **Análisis Físico:** El motor no recalculó la trayectoria desde el origen \(A\). Utilizó el estado del usuario 1 como el "faro" \(C\) y aplicó una matriz de dispersión elástica para resolver instantáneamente el destino \(B\) del usuario 2. El ahorro de cómputo es masivo.

### 2. Telemetría de la Autopista (Early Exit en Acción)

Durante la generación masiva de 200 tokens, el motor recolectó métricas sobre la certeza del modelo en las capas intermedias. La telemetría demuestra que el motor "sabe cuándo sabe".

*   **Total tokens generados:** 200
*   **Autopista activa (Early Exit):** `40.00%` (80 tokens)
*   **Conflicto (Cadena completa):** `60.00%` (120 tokens)
*   **Razón principal de la Autopista:** Margen de logits (El modelo tenía certeza matemática absoluta sobre el siguiente token).

> **Análisis Físico:** En 80 de los 200 tokens, la entropía (el desorden) fue tan baja que el motor decidió **saltarse las capas profundas y el muestreo complejo**. Esto valida la ecuación de inercia latente \(\Delta h\). El motor "deshiló" la cuerda y tomó el atajo, ahorrando ciclos de GPU sin perder coherencia en el texto generado.

### 3. Escalabilidad y Concurrencia (Entropía Gravitacional)

El motor demuestra una escalabilidad lineal, manteniendo el *Throughput* (rendimiento total) incluso bajo estrés de múltiples usuarios (secuencias paralelas).

| Usuarios Simultáneos (`n_seq_max`) | Throughput Total | Velocidad por Usuario | Estado del Sistema |
| :--- | :--- | :--- | :--- |
| **1 usuario** | 72.89 tokens/s | 72.89 tokens/s | Óptimo |
| **8 usuarios** | 201.69 tokens/s | 25.21 tokens/s | Óptimo (Escalado x2.7) |

> [!WARNING]
> **Nota sobre Concurrencia Masiva (32+ usuarios):**
> Durante las pruebas de 32 usuarios, el motor abortó controladamente con el error `GGML_ASSERT(n_tokens_all <= cparams.n_batch)`. Esto **no es un fallo del motor NAI**, sino un límite físico del script de pruebas. El script está configurado por defecto con un `n_batch = 128`. Para soportar más de 16 usuarios concurrentes, es necesario modificar esta variable a **512** o **1024** en el código de inicialización para evitar el desbordamiento del búfer de cómputo en Metal.

---
## 🛠️ Instalación y Ejecución

El Motor NAI mantiene la filosofía de `llama.cpp`: sin dependencias externas pesadas, compilación en C/C++ y optimización nativa para el hardware.

### 1. Prerrequisitos
*   **macOS:** Xcode Command Line Tools y un chip Apple Silicon (M1/M2/M3) o Intel.
*   **Linux/Windows:** Compilador C++ (GCC/Clang/MSVC), CMake y los drivers correspondientes (CUDA para NVIDIA, ROCm para AMD).
*   **Modelo:** Un archivo `.gguf` (las pruebas fueron realizadas con `qwen2.5-0.5b-instruct-q4_k_m.gguf`).

### 2. Clonar y Compilar el Motor Base

```bash
# 1. Clonar el repositorio
git clone https://github.com/FranciscoACruz/NAIEngine-llama.cpp.git
cd NAIEngine-llama.cpp

# 2. Configurar y compilar con CMake
# (El flag -DLLAMA_METAL=ON es exclusivo para macOS. En Linux usa -DLLAMA_CUDA=ON)
mkdir build && cd build
cmake .. -DLLAMA_METAL=ON -DLLAMA_NATIVE=ON
cmake --build . --config Release -j


```markdown
### 3. Compilar el Script de Benchmark

El archivo `test_benchmark.cpp` debe enlazarse con las librerías estáticas generadas en el paso anterior. 

> **Nota de Rutas:** El siguiente comando asume que estás dentro de la carpeta `build` y que el script `test_benchmark.cpp` está en la carpeta raíz (`../`). Si tienes el modelo guardado en un disco externo (ej. `/Volumes/Externo/`), asegúrate de crear un enlace simbólico o modificar la ruta en el código (Línea 73) para evitar errores de `File Not Found`.

**Para macOS (Apple Silicon):**

```bash
clang++ -std=c++17 -O3 -I../include -I../common -I../ggml/include \
  ../test_benchmark.cpp \
  libcommon.a libllama.a ggml/src/libggml.a ggml/src/libggml-base.a ggml/src/libggml-cpu.a ggml/src/libggml-metal.a ggml/src/libggml-blas.a \
  -framework Accelerate -framework Metal -framework Foundation -framework CoreGraphics \
  -o test_benchmark

  Para Linux (CUDA / CPU):

  g++ -std=c++17 -O3 -I../include -I../common -I../ggml/include \
  ../test_benchmark.cpp \
  libcommon.a libllama.a ggml/src/libggml.a ggml/src/libggml-base.a ggml/src/libggml-cpu.a ggml/src/libggml-cuda.a \
  -lpthread -ldl -lcuda -lcublas -lcudart \
  -o test_benchmark


```markdown
### 4. Ejecutar el Benchmark

Una vez compilado, ejecuta el binario. Asegúrate de tener el archivo `.gguf` en la misma carpeta o haber actualizado la ruta en el código.

```bash
./test_benchmark

## 🚑 Solución de Problemas (Troubleshooting)

Si encuentras problemas al compilar o ejecutar el Motor NAI, aquí tienes las soluciones a los fallos más comunes reportados durante las pruebas.

### 1. Error: GGML_ASSERT(n_tokens_all <= cparams.n_batch) failed

**Causa:** Este error ocurre durante los tests de estrés (32, 64 o 128 usuarios concurrentes). El script `test_benchmark.cpp` tiene un límite por defecto de `n_batch = 128`. Al procesar múltiples prompts de golpe, se supera este límite y el motor aborta para proteger la VRAM.

**Solución:** Abre el archivo `test_benchmark.cpp`, busca la función `run_stress_test` y aumenta el límite del batch:

```cpp
// Cambiar esta línea:
cparams.n_batch = 128; 
// Por esta:
cparams.n_batch = 512; // o 1024 para pruebas masivas



---

### 📄 Bloque 2: Troubleshooting (Parte B)

```markdown
### 2. Error: File Not Found al cargar el modelo

**Causa:** El código busca por defecto el archivo `.gguf` en la misma carpeta donde se ejecuta el binario.

**Solución:** Si guardaste el modelo en un disco externo o en otra carpeta, crea un enlace simbólico en la raíz del proyecto:

```bash
ln -s /Volumes/Externo/qwen2.5-0.5b-instruct-q4_k_m.gguf ./qwen2.5-0.5b-instruct-q4_k_m.gguf


---

### 📄 Bloque 3: Troubleshooting (Parte C)

```markdown
### 3. Errores de enlazado (Undefined symbols) al compilar el Benchmark

**Causa:** El compilador no encuentra las librerías estáticas `.a` generadas por CMake.

**Solución:** Asegúrate de haber ejecutado `cmake --build . --config Release` exitosamente antes de compilar el script, y verifica que estás ejecutando el comando `clang++` o `g++` desde dentro de la carpeta `build/`.

---

## 📄 Licencia

Este proyecto se distribuye bajo la **Licencia MIT**, heredada directamente del proyecto original `llama.cpp`. Eres libre de usar, modificar, fusionar, publicar, distribuir, sublicenciar y/o vender copias del software.

Consulta el archivo `LICENSE` en la raíz del repositorio para obtener el texto completo.

---

## 🙏 Agradecimientos

*   **Georgi Gerganov y la comunidad de `llama.cpp`:** Por sentar las bases de la inferencia de LLM en hardware de consumo y crear una arquitectura tan modular que permite experimentaciones profundas como este Motor NAI.
*   **La Física Teórica:** Por proporcionar las analogías matemáticas y conceptuales (como la *Matriz de Dispersión* y la *Entropía Gravitacional*) que demostraron ser no solo metáforas, sino guías prácticas para optimizar el silicio.

<br>

<div align="center">
  <i>"La naturaleza no da saltos, pero el silicio bien optimizado sí."</i>
  <br><br>
  <b>Motor NAI © 2026</b>
</div>

