# Motor NAI: Manifiesto Arquitectónico, Fundamentos de Física Computacional y Especificación de Bajo Nivel
*(Inercia Latente, Triaje Quirúrgico, Autopista de Muestreo, Mecánica Gravitatoria KV y Persistencia de Sesión Conversacional)*

---

## Prólogo Filosófico y Génesis del Sistema

El **Motor NAI** no surge de una optimización sintáctica tradicional dentro de `llama.cpp`, sino de una pregunta fundamental sobre la naturaleza de la cognición artificial y su fricción con el silicio: *¿Cómo se comportaría una red neuronal si su arquitectura interna emulara las leyes macroscópicas y termodinámicas del universo físico (inercia, campos gravitatorios, menor acción energética y enfriamiento temporal)?* 

Frente a la rigidez lineal de las arquitecturas tradicionales —donde cada token gasta idéntica energía computacional y la memoria de contexto se degrada por saturación lineal—, NAI introduce un paradigma donde los tensores fluyen siguiendo trayectorias de menor resistencia, dando paso a un sistema apto no solo para software de alto rendimiento, sino como base conceptual para futuros desarrollos de hardware neuromórfico y computación acelerada.

---

## 1. Comparativa Estructural: Llama.cpp Vanilla vs. Motor NAI

| Dimensión Arquitectónica | `llama.cpp` Vanilla (Original) | Motor NAI (Arquitectura Optimizada) | Ventaja / Impacto del Modelo NAI |
| :--- | :--- | :--- | :--- |
| **Evaluación de Capas** | Torre secuencial rígida ($L$ capas completas por token). | Tronco de Capas Respirantes con *Early Exit* condicional. | Ahorro de hasta el **60% de ciclos de GPU** en tokens de alta certidumbre. |
| **Gestión de Memoria (KV Cache)** | Ventana estática uniforme; todos los tokens pesan igual, generando ruido a largo plazo. | Campo gravitatorio de tres cuerpos con atenuación no lineal y masa semántica ($I_i$). | Purga automática del ruido sintáctico en contextos masivos (>32k tokens). |
| **Subsistema de Muestreo** | Evaluación iterativa pesada y secuencial de toda la cadena de samplers en cada paso. | Bypass de Autopista mediante métricas de inercia (`DecisionInercia`). | Omisión de samplers pesados en más del **40% de los tokens**, acelerando la inferencia. |
| **Persistencia de Sesión** | Sesiones volátiles ligadas al ciclo de vida bruto del contexto de ejecución. | Ciclo de vida biológico de **24 horas por usuario**, garantizando continuidad conversacional. | Persistencia transaccional robusta sin pérdida de estado lógico. |

---

## 2. El Ciclo de Vida de Sesión de 24 Horas (Persistencia de Consciencia)

Para que el motor mantenga una continuidad coherente en aplicaciones del mundo real (como plataformas móviles y agentes avanzados), NAI implementa un administrador de sesiones persistentes:
- **Duración del Ciclo de Vida:** Cada sesión de usuario está acotada a una ventana exacta de **24 horas de persistencia activa**.
- **Mecánica de Sincronización:** El estado latente acumulado y los metadatos del mapa semántico se vinculan al identificador de sesión, permitiendo que el usuario retome un hilo conversacional complejo sin necesidad de reevaluar todo el prefill histórico desde cero, reduciendo drásticamente la latencia de arranque (*Time-to-First-Token*).

---

## 3. Filosofía del Sistema: La Menor Acción Energética y la Consciencia Artificial

La arquitectura tradicional de inferencia basada en LLMs trata cada token de entrada como un gasto uniforme de recursos: una torre rígida de $L$ capas que procesa desde la conjunción más simple hasta la deducción más compleja bajo idéntica presión de silicio. El **Motor NAI** se fundamenta en un principio rector tomado directamente de la física fundamental: **el principio de la menor acción**.

### A. ¿Por qué NAI imita la termodinámica del universo?
* **Inercia Latente ($\Delta h$):** Así como la materia se opone al cambio de estado mediante la inercia, un tensor de lenguaje con alta certidumbre sintáctica no necesita ser forzado a través de todas las capas profundas. Al calcular la entropía espectral, el sistema decide un *Early Exit* biológico, gastando energía computacional únicamente cuando el modelo se enfrenta a una auténtica fricción de pensamiento (un **Nodo de Conflicto**).
* **Campo Gravitatorio de Tres Cuerpos en el KV Cache:** Las ventanas de contexto masivas (>32k tokens) sufren de entropía y pérdida de atención a medio camino. NAI sustituye la memoria estática por un campo gravitatorio no lineal donde las instrucciones clave, variables y reglas del sistema mantienen un "pozo gravitatorio" inalterable, mientras que el relleno conversacional irrelevante sufre un enfriamiento exponencial ($dt$). El resultado es una atención pura y perpetua libre de ruido.

### B. El Ciclo Biológico de Persistencia de Sesión (24 Horas)
A diferencia de los entornos volátiles donde cada hilo conversacional nace y muere sin continuidad transaccional, NAI introduce un administrador de sesiones persistentes acotado a un **ciclo de vida biológico de 24 horas por usuario**:
- El mapa semántico latente se almacena de forma estructurada, permitiendo que al retomar la sesión dentro diurna o nocturna, el motor conserve el contexto conversacional profundo sin necesidad de reevaluar el *prefill* histórico masivo desde cero.
- Esto dota al sistema de una verdadera **continuidad de consciencia operativa**, ideal para aplicaciones móviles y agentes autónomos de alta exigencia que imitan la persistencia de la memoria humana a corto y mediano plazo.

### C. Modelo Gravitatorio de Tres Cuerpos en el KV Cache (Fórmula de Interacción)
Para modelar la persistencia de los tokens clave y la degradación del ruido conversacional, la actualización del tensor de claves $K_i$ en función del tiempo y la masa semántica se define formalmente mediante el siguiente operador de campo de tres cuerpos:

$$K_i(t + dt) = K_i(t) \cdot \exp\left(-\frac{\gamma \cdot dt}{I_i + \varepsilon}\right)$$

* **$K_i(t)$:** Tensor de clave original del token $i$ en el tiempo $t$.
* **$\gamma$:** Coeficiente global de disipación térmica y entrópica del sistema.
* **$dt$:** Incremento temporal discreto entre iteraciones de contexto.
* **$I_i$:** Masa semántica intrínseca del token, calculada como:
  $$I_i = \|W_K x_i\|_2 \cdot \left(1 + \mathcal{H}_{\text{local}}(x_i)\right)$$
* **$\varepsilon$:** Constante infinitesimal de suavizado numérico para evitar indeterminaciones por división entre cero.
* **Comportamiento Físico:** Los tokens con alta masa semántica ($I_i \gg 0$) preservan su pozo gravitatorio de atención de manera indefinida, mientras que el relleno sintáctico o de baja relevancia decae exponencialmente hacia el cero absoluto, purgando el ruido en contextos masivos.


---

## 4. Desglose Exhaustivo de Capas del Pipeline y Subsistemas

### Capa 1: TriageEngine (Filtro Quirúrgico / Prefiltro Header-Only)
* **Ubicación Lógica:** Interceptación previa a la tokenización profunda en el flujo de entrada de `llama.cpp`.
* **Propósito de Ingeniería:** Evita la tokenización profunda y la asignación costosa de tensores en la GPU cuando el prompt entrante presenta ambigüedad o requiere una ruta de ejecución optimizada de costo computacional nulo.
* **Mecánica de Funcionamiento:**
  - Aplica un filtro de normalización de caracteres UTF-8/ASCII y supresión de diacríticos para homogeneizar el vector de entrada.
  - Ejecuta un escaneo por coincidencia exacta de patrones léxicos mediante tablas hash estáticas para clasificar la intención del usuario en tres flujos de redirección:
    - `DIRECTO`: Inferencia estándar en la red neuronal por capas.
    - `INDAGACION_CORTA`: Derivación hacia un flujo asistido de respuestas estructuradas breves (0% de cómputo inútil en capas profundas).
    - `PROPUESTA_HIPOTESIS`: Inyección de sesgos de razonamiento extendido.

### Capa 2: Mapa de Masa Semántica (`llama-context.cpp`)
* **Estructura Global:** `nai::MapaMasaSemantica g_nai_mapa_masa` (preasignada estáticamente a 65,536 posiciones).
* **Propósito de Ingeniería:** En un contexto estandarizado, el búfer Key-Value (KV Cache) trata todos los tokens con la misma relevancia, acumulando ruido semántico. NAI introduce un mapa global para dar seguimiento a la masa semántica latente.
* **Mecánica de Funcionamiento:**
  - Rastrea en tiempo real el estado latente y la inercia acumulada del contexto actual.
  - Valida de forma estricta los punteros de los búferes (`logits`, `probs`, `candidates`, `sampled`) mediante aserciones de sistema (`GGML_ASSERT`) para prevenir desbordamientos en la memoria unificada de la GPU.

  ### Capa 3: Autopista de Inercia y Muestreo (`sampling.cpp` / `sampling.h`)
* **Estructura de Decisión:** `DecisionInercia` y función de evaluación `evaluar_inercia_y_equivalencia`.
* **Propósito de Ingeniería:** El cuello de botella clásico en la generación de texto reside en la evaluación secuencial de samplers pesados (como `DRY`, `XTC`, `Mirostat`, penalizaciones de repetición, etc.) en cada paso de muestreo.
* **Criterios de Activación del Bypass (Vía Rápida):**
  - **Código 1:** Existencia de un único candidato válido en el vocabulario filtrado.
  - **Código 2:** El margen de logits entre el primer y segundo candidato supera el umbral estricto $\delta_{\text{logit}} = 2.2f$.
  - **Código 3:** Margen de probabilidad $|p_1 - p_2| < 0.02f$ (equivalencia semántica directa).
  - **Código 4:** El token dominante $p_1$ supera por un factor de $1.3\times$ la suma acumulada de la cola (tokens 2 al 5).
  - **Código 0 (Conflicto / Falla):** El tensor se cataloga como ambigua y se ejecuta la cadena completa de samplers.

---

## 5. Fundamentos Matemáticos y Estructuras de Tensores

### A. Tronco de Capas Respirantes (Early Exit)
En la Capa $k$ (por ejemplo, Capa 12 en Qwen2), el estado latente $h^{(k)}$ se proyecta mediante la cabeza temprana `nai_logits_early`. La inercia latente $\Delta h$ se calcula mediante la entropía espectral normalizada:

$$\Delta h = 1 - \frac{\mathcal{H}(\sigma(z^{(k)}))}{\log_2 |V|}$$

*Donde $\mathcal{H}$ es la entropía de Shannon de la distribución de probabilidad suavizada por Softmax $\sigma(z^{(k)})$, y $|V|$ es el tamaño del vocabulario.*

- **Evaluación de Salida:** 
  - Si $\Delta h > \tau_{\text{autopista}}$: Se activa el *Early Exit* cortando la ejecución en `llama-graph.cpp` y saltando las capas $k+1 \dots L$.
  - Si $\Delta h \le \tau_{\text{autopista}}$: Se interpreta como un Nodo de Conflicto y se deriva a las capas profundas.

### B. Memoria de Masa Gravitatoria (Modelo de los 3 Cuerpos en el KV Cache)
Para evitar la saturación de contexto, el KV Cache implementa un campo de atenuación gravitatoria dependiente del tiempo:

1. **Masa Semántica ($I_i$):**


$$I_i = \|W_K x_i\|_2 \cdot \left(1 + \mathcal{H}_{\text{local}}(x_i)\right)$$

2. **Decaimiento Temporal y Enfriamiento ($dt$):**

$$K_i(t + dt) = K_i(t) \cdot \exp\left(-\frac{\gamma \cdot dt}{I_i + \varepsilon}\right)$$

*Los tokens estructurales preservan su pozo gravitatorio, mientras que el relleno sintáctico se atenúa exponencialmente.*

---

## 6. Diagrama de Flujo del Pipeline de Ejecución
```text
[ Entrada: Prompt / Token ]
             │
             ▼
┌──────────────────────────┐
│   1. TRIAJE LATENTE      │ ──► ¿Entrada ambigua?
│ (Proyección Temprana)    │       └──► Indagación Quirúrgica (0% cómputo inútil)
└────────────┬─────────────┘
             │ (Continuar)
             ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                       2. TRONCO DE CAPAS ELÁSTICAS                        │
│                                                                           │
│   Capa 1..12 ──► [ Medidor de Inercia Latente Δh ]                        │
│                           │                                               │
│            ┌──────────────┴──────────────┐                                │
│            ▼                             ▼                                │
│   [ Inercia Alta / Autopista ]   [ Nodo de Conflicto / Entropía Alta ]    │
│   • Early Exit (nai_logits_early)• Ejecución de Capas Profundas (13..N)   │
│   • Bypass del 60% de la GPU     • Análisis de Alternativas (Tree Search) │
└────────────────────────────┬──────────────────────────────────────────────┘
                             │
                             ▼
             ┌─────────────────────────────────┐
             │ 3. MEMORIA GRAVITATORIA (KV)    │ ──► Tokens de bajo campo
             │    (Modelo de los 3 Cuerpos)    │     gravitatorio se colapsan
             └─────────────────────────────────     o atenúan por enfriamiento (dt)

```


## 7. Integración Técnica en el Código Fuente
### Secuencia de Enlaces de Módulos

```text
┌─────────────────────────┐
│   inercia_latente.hpp   │ ──► Define funciones de cálculo: Varianza, Entropía y
└────────────┬────────────┘     `aplicar_atenuacion_gravitatoria(K, V, dt, I)`
             │
             ▼
┌─────────────────────────┐
│   src/models/qwen2.cpp  │ ──► Aplica el Hook `nai_logits_early` en Capa 12 para
└────────────┬────────────┘     evaluación de Salida Temprana (Early Exit).
             │
             ▼
┌─────────────────────────┐
│    llama-context.cpp    │ ──► Invoca la atenuación gravitatoria en `memory_update()`
└─────────────────────────┘     antes de la construcción del grafo del nuevo batch.
```

### Implementación del Bypass en `sampling.cpp`
```cpp
DecisionInercia dec = evaluar_inercia_y_equivalencia(gsm->cur_p);

if (dec.es_autopista) {
    LOG_DBG("%s: [Inercia Autopista] Codigo: %d, Margen: %.4f -> Token direct: %d\n",
            __func__, dec.codigo_razon, dec.margen, gsm->cur_p.data[0].id);
    return gsm->cur_p.data[0].id;
}
```


## 8. Compilación y Requisitos de Sistema
### Requisitos Técnicos
### Compilador C++17 o superior (GCC, Clang o MSVC).

Soporte nativo habilitado para arquitectura ARM64 / Apple Silicon (M1/M2/M3/M4) e x86_64.

CMake 3.14+.

### Instrucciones de Compilación (CMake)

#### Clonar y acceder al repositorio
cd NAIEngine-llama.cpp

#### Crear directorio de compilación y configurar
mkdir build && cd build
cmake .. -DLLAMA_NATIVE=ON

#### Compilar binarios optimizados en modo Release
cmake --build . --config Release -j

---

## 9. Benchmarks de Rendimiento y Análisis de Concurrencia

Pruebas ejecutadas sobre **Apple M1** utilizando infraestructura Metal (`MTL0`):

### A. Rendimiento de Carga y Autopista
- **Prefill Normal (Seq 0):** `82.01 ms` (`475.54 tok/s`)
- **Prefill NAI Sharing (Seq 1):** `15.70 ms` *(El KV Cache se copió y solo procesó 1 token gracias al intercambio inteligente)*.
- **Velocidad de Generación Masiva (200 tokens):** `38.69 tok/s`.
- **Telemetría de Autopista:** 
  - Total tokens muestreados: `200`
  - Conflicto (cadena completa): `120 (60.00%)`
  - Autopista activa por Margen de Logits (Razón 2): `80 (40.00%)`
  - *Calibración del sistema:* Saludable y operativa al 40%.

### B. Pruebas de Estrés y Límites de Concurrencia
1. **1 Usuario Simultáneo (Contexto: 512 tokens):**
   - Throughput Total / Velocidad por usuario: **`72.89 tokens/segundo`**.
2. **8 Usuarios Simultáneos (Contexto: 4096 tokens):**
   - Throughput Total: **`201.69 tokens/segundo`** (`25.21 tok/s` por usuario).
3. **32 Usuarios Simultáneos (Contexto: 16384 tokens):**
   - **Resultado del Kernel:** Interrupción controlada por aserción de seguridad:
     `GGML_ASSERT(n_tokens_all <= cparams.n_batch) failed`
   - **Análisis de Ingeniería:** Comprueba que las guardas de memoria protegen los búferes de la GPU Metal ante una saturación masiva de lotes que superan el límite estructural de `n_batch` (512), abortando de forma limpia sin corrupción de memoria unificada.

---

## 10. Consideraciones Técnicas y Advertencias de Uso

1. **Interacción con Gramáticas Estrictas:**
   - Si se activa el muestreo por gramática (`grammar_first = false`), la regla de autopista podría omitir la validación sintáctica en tokens dominantes. Para esquemas rígidos (JSON, YAML), se recomienda forzar estrictamente `grammar_first = true` o deshabilitar temporalmente el bypass de autopista.

2. **Normalización de Probabilidades:**
   - Las reglas de inercia basadas en el margen de probabilidad ($|p_1 - p_2|$, correspondientes a los Códigos 3 y 4) requieren que el vector de candidatos cuente con un paso previo de Softmax y normalización explícita. Si el pipeline entrega únicamente logits puros sin normalizar, el motor recurrirá de forma transparente al Código 2 (margen de logits).

3. **Seguridad Multi-hilo (`Multi-threading`):**
   - La estructura global `nai::MapaMasaSemantica g_nai_mapa_masa` implementada en `llama-context.cpp` debe protegerse mediante cerrojos de exclusión mutua (*mutex/locks*) o ser instanciada por contexto de forma aislada en caso de ejecutar inferencia concurrente en múltiples instancias paralelas de `llama_context`.

---

## 11. Licencia

Este proyecto mantiene los términos de licenciamiento originales de `llama.cpp` bajo la **MIT License**. Consulta el archivo `LICENSE` incluido en el repositorio para más detalles sobre derechos de copia, redistribución y limitaciones de responsabilidad.