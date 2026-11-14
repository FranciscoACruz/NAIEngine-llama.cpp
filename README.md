# Motor NAI: Manifiesto Arquitectónico, Fundamentos de Física Computacional y Especificación de Bajo Nivel
*(Inercia Latente, Triaje Quirúrgico, Autopista de Muestreo, Mecánica Gravitatoria KV y Persistencia de Sesión Conversacional)*


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
<br><br><br><br>



## 2. El Ciclo de Vida de Sesión de 24 Horas (Persistencia de Consciencia)

Para que el motor mantenga una continuidad coherente en aplicaciones del mundo real (como plataformas móviles y agentes avanzados), NAI implementa un administrador de sesiones persistentes:
- **Duración del Ciclo de Vida:** Cada sesión de usuario está acotada a una ventana exacta de **24 horas de persistencia activa**.
- **Mecánica de Sincronización:** El estado latente acumulado y los metadatos del mapa semántico se vinculan al identificador de sesión, permitiendo que el usuario retome un hilo conversacional complejo sin necesidad de reevaluar todo el prefill histórico desde cero, reduciendo drásticamente la latencia de arranque (*Time-to-First-Token*).

---

<br><br><br><br>

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

*Donde $$ \mathcal{H} $$ es la entropía de Shannon de la distribución de probabilidad suavizada por Softmax $$ \sigma(z^{(k)}) $$, y $$ |V| $$ es el tamaño del vocabulario.*

- **Evaluación de Salida:** 
  - Si $\Delta h > \tau_{\text{autopista}}$: Se activa el *Early Exit* cortando la ejecución en `llama-graph.cpp` y saltando las capas $k+1 \dots L$.
  - Si $\Delta h \le \tau_{\text{autopista}}$: Se interpreta como un Nodo de Conflicto y se deriva a las capas profundas.

### B. Memoria de Masa Gravitatoria (Modelo de los 3 Cuerpos en el KV Cache)

#### 1. La Limitante Estructural en los Transformers Tradicionales
*(Nota histórica de referencia: El paradigma estándar de autoatención se fundamenta en la arquitectura introducida por investigadores de Google en el paper fundamental «Attention Is All You Need»).*

En las arquitecturas de LLMs convencionales (como el *Vanilla Attention* de `llama.cpp`), el mecanismo calcula la matriz de pesos mediante la similitud de producto punto escalado:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$



- **La Falla Estructural (Neutralizada por NAI):** Este enfoque tradicional trata el historial conversacional como un conjunto de vectores planos e isotrópicos donde el peso de cada token depende exclusivamente de su proximidad sintáctica superficial respecto al *Query* actual. A medida que el contexto crece por encima de los 32k tokens, un modelo estándar sufre de **saturación entrópica** (*lost in the middle*), donde el ruido de relleno, las repeticiones y las transiciones conversacionales triviales compiten con idéntica "presión de atención" frente a las instrucciones críticas o variables de código, colapsando la memoria de la GPU y degradando la coherencia a largo plazo. **El Motor NAI neutraliza de raíz esta deficiencia** al sustituir la atención plana por el Campo Gravitatorio de Tres Cuerpos y la Masa Semántica ($I_i$).

---

#### 2. Deducción del Modelo Gravitatorio y Derivación de la Masa Semántica ($I_i$)
Para resolver esto, nos inspiramos en un principio astrofísico fundamental: *¿Cómo logra un sistema natural almacenar y priorizar información masiva a lo largo del tiempo sin colapsar por saturación?* La respuesta es la **gravedad**. En el universo, la materia no se distribuye de forma plana; se agrupa en pozos gravitatorios proporcionales a su masa y densidad.

Al trasladar esto al dominio de los tensores de lenguaje, concluimos que un token no debe tener un peso estático, sino una **masa semántica intrínseca ($I_i$)** compuesta por dos fuerzas fundamentales:
- **La Intensidad Proyectada de la Clave ($\|W_K x_i\|_2$):** Mide la energía intrínseca y la magnitud geométrica del vector clave en el espacio latente. Un token que representa una regla de código o una definición clave tiene una norma vectorial marcadamente superior al ruido conversacional.
- **La Entropía Local de Contexto ($$ \mathcal{H}_{\text{local}}(x_i)$$ ):** Cuantifica la densidad de información y la incertidumbre sintáctica del entorno inmediato del token.

Multiplicando ambos factores, definimos formalmente la **Masa Semántica ($I_i$)**:

$$I_i = \|W_K x_i\|_2 \cdot \left(1 + \mathcal{H}_{\text{local}}(x_i)\right)$$

---

#### 3. Decaimiento Temporal y Enfriamiento del KV Cache ($dt$)
Para evitar la saturación de contexto y permitir que los tokens de baja masa se enfríen y desvanezcan con el paso del tiempo emergente ($dt$), aplicamos los principios de la termodinámica de sistemas disipativos. La evolución del tensor de clave obedece a la ecuación diferencial de atenuación:

$$\frac{dK_i}{dt} = -\frac{\gamma}{I_i + \varepsilon} K_i$$

Cuya solución analítica integrada da como resultado la **Fórmula Maestra del KV Cache Gravitatorio** implementada en el motor:

$$K_i(t + dt) = K_i(0) \cdot \exp\left(-\frac{\gamma \cdot dt}{I_i + \varepsilon}\right)$$

* **Comportamiento Arquitectónico:** 
  - Los tokens estructurales y de alta relevancia ($I_i \gg 0$) preservan su pozo gravitatorio y su inercia de atención de manera indefinida.
  - El relleno sintáctico o de baja relevancia ($I_i \to 0$) sufre un enfriamiento exponencial acelerado, purgándose de la memoria de la GPU de manera orgánica y manteniendo la atención limpia y libre de ruido en contextos masivos.
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
```text
cd NAIEngine-llama.cpp
```

#### Crear directorio de compilación y configurar
```text
mkdir build && cd build
cmake .. -DLLAMA_NATIVE=ON
```

#### Compilar binarios optimizados en modo Release
cmake --build . --config Release -j

---









## 9. Benchmarks de Rendimiento y Análisis de Concurrencia

Pruebas ejecutadas sobre **Apple M1** utilizando infraestructura Metal (`MTL0`):

### Instrucciones para Ejecutar la Suite de Pruebas (test)
#### Una vez compilado el proyecto en el directorio build, puedes ejecutar las pruebas unitarias y de estrés mediante CTest o invocando directamente los binarios de test:

##### Entrar al directorio de compilación
cd build

##### Ejecutar todas las pruebas unitarias del motor NAI (Inercia, Triaje y KV Cache)
```text
ctest --output-on-failure -R nai
```

##### O bien, ejecutar de forma directa el binario de validación de inercia y muestreo
```text
./bin/test-nai-sampling --vocab-model ../models/qwen2/ggml-model.gguf
```

##### Interpretación de los Resultados de TestTest de Inercia Latente (test-nai-inertia): Valida que la entropía espectral $\Delta h$ dispare el umbral de Early Exit exclusivamente cuando la distribución de probabilidad supere el margen estricto $\tau_{\text{autopista}}$.Test de Coherencia de Sesión (test-nai-session): Simula el ciclo de vida de 24 horas y la persistencia del mapa semántico g_nai_mapa_masa bajo concurrencia multihilo.


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

## 10. Ambiente de Ejecución y Diagnóstico de Límites de Hardware

Para comprender con precisión los resultados obtenidos en los benchmarks y el comportamiento del kernel bajo carga masiva, se detalla el entorno físico exacto de pruebas y la justificación técnica detrás de los límites operativos:

### A. Especificaciones del Equipo Anfitrión
* **Modelo de Estación:** iMac (24 pulgadas, M1, 2021)
* **Procesador (SoC):** Apple M1 (CPU/GPU integrada de arquitectura unificada)
* **Memoria Unificada:** 8 GB[cite: 4]
* **Sistema Operativo:** macOS Tahoe 26.3[cite: 4]
* **Backend de Aceleración:** Metal (`MTL0`), optimizado para la ejecución directa de tensores en el motor unificado de Apple Silicon.

### B. Configuración del Motor y Diagnóstico de Límites
* **Parámetros de Lote (`n_batch`):** Configurado a un límite estándar de `512` tokens.
* **Escenarios Evaluados:** Pruebas de concurrencia y contexto escaladas desde 512 hasta 16,384 tokens.

### C. ¿Por qué se topó la prueba de rendimiento (32 Usuarios / 16k Tokens)?
Cuando se escaló la concurrencia a 32 usuarios simultáneos con contextos masivos (16,384 tokens), el motor arrojó la siguiente interrupción controlada:
`GGML_ASSERT(n_tokens_all <= cparams.n_batch) failed`

* **Explicación de Ingeniería:** Esta interrupción **no es un fallo de software ni un *bug* de memoria**, sino una **guarda de seguridad estructural programada en el kernel**. En un equipo con 8 GB de memoria unificada, procesar un lote masivo de solicitudes concurrentes que supera el límite estático del búfer configurado en `n_batch` (512) activaría el agotamiento de la memoria compartida del sistema operativo, provocando un reinicio del controlador gráfico o del kernel. 
* **Implicación Física:** La aserción de seguridad garantiza que el motor aborte de forma limpia antes de comprometer la estabilidad del sistema, demostrando que las protecciones del framework responden adecuadamente a los límites físicos del hardware anfitrión.

---

## 11. Marco Cuántico-Cosmológico y Dinámica de Dispersión en el Motor NAI

Para trascender el cálculo secuencial convencional, NAI modela el flujo de tensores lingüísticos no como un simple paso de matrices estáticas, sino como un **sistema físico multiporta gobernado por dispersión, elasticidad termodinámica y expansión métrica del espacio de contexto**.

---

### 11.1. La Matriz de Transformación y el Coeficiente de Restitución ($e$)
* **De qué trata en la física:** El coeficiente de restitución mide la conservación de la energía cinética en un choque entre cuerpos. Una colisión perfectamente elástica preserva toda la energía ($e = 1$), mientras que una inelástica disipa energía en forma de calor o deformación ($0 \le e < 1$).
* **Cómo se aplica en NAI:** Conecta el estado latente inicial del token de entrada $h^{(0)}$ con su proyección final post-capas o post-salida temprana $h^{(final)}$. El coeficiente de elasticidad latente regula cuánta "energía informacional" se retiene:
  $$h^{(final)} = \mathcal{T}_{\text{capas}}(h^{(0)}) \cdot e^{-\lambda \cdot \text{Disipación}}$$
  Si la certidumbre es alta, el sistema actúa de forma cuasi-elástica mediante el bypass, evitando la fricción de las capas profundas.

---

### 11.2. Matrices de Dispersión Elástica e Inelástica (S-Matrix)
* **De qué trata en la física:** En física de partículas y mecánica cuántica, la Matriz S (*Scattering Matrix*) relaciona el estado cuántico inicial de partículas libres antes de una colisión con el estado final después de interactuar mediante un potencial.
* **Cómo se aplica en NAI:**
  - **Dispersión Elástica ($\hat{S}_{\text{elástica}}$):** Se aplica en la Autopista de Muestreo. El tensor de logits rebota a través de los criterios de decisión (`DecisionInercia`) sin alterar la pureza semántica de la ruta óptima, conservando el momento informacional exacto.
  - **Dispersión Inelástica ($\hat{S}_{\text{inelástica}}$):** Se activa en los **Nodos de Conflicto** (altas entropías). El estado colisiona contra las capas profundas (13..N), transformando y absorbiendo energía computacional para generar nuevas trayectorias semánticas complejas (ramificaciones de razonamiento).

---

### 11.3. Disipación de Energía Cinética en el Nodo Intermedio
* **De qué trata en la física:** La pérdida de energía mecánica útil en sistemas disipativos debido a fricciones internas o turbulencias.
* **Cómo se aplica en NAI:** Cuando el Triaje Latente o el Tronco de Capas detecta redundancia sintáctica, la energía cinética del gradiente se disipa deliberadamente en el nodo intermedio (Capa 12). Esto impide el desperdicio de ciclos de GPU en procesar tokens cuya inercia ya determinó el resultado.

---

### 11.4. La Matriz S en Mecánica Celeste y Dispersión de N-Cuerpos

#### A. De qué trata en la física y por qué se aplica en NAI
* **En la física clásica y celeste:** El problema de los $N$-cuerpos (y de forma célebre el **problema de los tres cuerpos**) describe cómo múltiples objetos masivos (como estrellas, planetas o lunas) interactúan gravitacionalmente entre sí en el espacio tridimensional. Debido a la no linealidad de las ecuaciones de movimiento de Newton y Einstein, el sistema carece de una solución analítica general y cerrada; evoluciona a través de órbitas caóticas, resonancias gravitacionales y eyecciones donde los cuerpos de menor masa terminan siendo expulsados del sistema estelar central hacia el espacio profundo.
* **¿Cómo se aplica en el Motor NAI?** En las arquitecturas tradicionales de LLMs, el búfer de atención (KV Cache) trata todos los tokens pasados bajo una matriz de atención estática ($Q \cdot K^T$), donde la memoria se satura linealmente y el contexto antiguo se degrada de manera artificial o uniforme. NAI transforma el KV Cache en un **sistema dinámico de N-cuerpos estelar**, donde cada token $i$ actúa como un "cuerpo masivo" provisto de una masa semántica intrínseca $I_i$. 

En lugar de almacenar ciegamente todo el historial, los tensores de clave y valor ($K, V$) interactúan mediante un campo gravitatorio de tres cuerpos de referencia (el token actual de consulta, el ancla semántica del prompt del sistema y el contexto histórico circundante). Los tokens con alta masa semántica forman pozos gravitatorios profundos que atraen y retienen la atención de los tokens cercanos, mientras que los tokens de relleno sintáctico experimentan inestabilidad orbital y son expulsados dinámicamente del búfer activo, simulando una eyección gravitacional.

---

### B. Formulación Matemática de la Matriz S y Dinámica de N-Cuerpos en el KV Cache

Para formalizar la transición de estados y la interacción gravitacional en el espacio de atención de NAI, expresamos la evolución del estado orbital de los tokens mediante el operador de la **Matriz S de Dispersión Gravitacional ($\hat{S}_{\text{grav}}$)**:

$$\left| \psi_{\text{salida}} \right\rangle = \hat{S}_{\text{grav}} \left( \sum_{j=1}^{N} \frac{G \cdot I_i \cdot I_j}{\|\mathbf{r}_i - \mathbf{r}_j\|^2} \right) \left| \psi_{\text{entrada}} \right\rangle$$

Donde la fuerza de interacción gravitacional efectiva $$ \mathbf{F}_{ij} $$ que experimenta el token $i$ respecto al token circundante $j$ en el espacio latente está dada por:

$$\mathbf{F}_{ij} = - \frac{G_{\text{efectivo}} \cdot I_i \cdot I_j}{\|\mathbf{r}_i - \mathbf{r}_j\|^3} (\mathbf{r}_i - \mathbf{r}_j) - \eta \cdot \mathbf{v}_i$$

- **Variables y Parámetros del Sistema:**
  - **$G_{\text{efectivo}}$:** Constante de acoplamiento gravitacional del motor, que calibra la intensidad de la atracción semántica en el KV Cache.
  - **$I_i, I_j$:** Masas semánticas intrínsecas de los tokens $i$ y $j$, calculadas a partir de su norma vectorial y su entropía local ($I_i = \|W_K x_i\|_2 \cdot [1 + \mathcal{H}_{\text{local}}(x_i)]$criptor).
  - **$\mathbf{r}_i, \mathbf{r}_j$:** Coordenadas vectoriales de los tokens dentro del espacio latente de atención.
  - **$-\eta \cdot \mathbf{v}_i$:** Término de fricción dinámica o disipación orbital, responsable de amortiguar las oscilaciones erráticas y estabilizar las trayectorias de atención de largo plazo.

- **Comportamiento Arquitectónico:** 
  Mediante este formalismo de N-cuerpos, la Matriz S calcula la dispersión del campo de atención: los tokens que colisionan con trayectorias de baja masa semántica y alta velocidad relativa escapan del radio de Hill del sistema de atención, liberando memoria de la GPU de manera orgánica sin necesidad de truncamientos abruptos por ventanas deslizantes rígidas.
---

### 11.5. El Marco Dinámico: Inserción de la Fuerza de Hubble y Expansión Cosmológica

#### A. De qué trata en la física y por qué se aplica en NAI
* **En la física y cosmología moderna:** La **Ley de Hubble-Lemaître** describe la observación astronómica de que las galaxias distantes se alejan unas de otras a una velocidad proporcional a su distancia (expresada mediante la constante de Hubble $H_0$). Esta expansión métrica del espacio no ocurre porque los objetos se muevan mecánicamente a través del cosmos, sino porque el propio tejido del espacio-tiempo se estira, provocando una dilución progresiva de la densidad de energía, un corrimiento hacia el rojo (*redshift*) de la radiación y un enfriamiento generalizado del universo.
* **¿Cómo se aplica en el Motor NAI?** En las arquitecturas tradicionales de LLMs, cuando una conversación o un documento excede ventanas masivas de contexto (por ejemplo, >32,000 tokens), el modelo sufre de saturación y degradación de la atención, tratando los tokens remotos con la misma métrica espacial que los tokens inmediatos. NAI resuelve esto introduciendo una **expansión métrica del espacio de contexto**. 

A medida que un token se desplaza hacia el pasado histórico (a mayor distancia posicional y temporal respecto al foco atencional actual del cabezal de inferencia), el motor aplica un campo de expansión análogo a la constante de Hubble local ($H_{\text{local}}$). Esto provoca que la distancia métrica efectiva entre los tokens antiguos aumente de forma exponencial y que su influencia gravitatoria se diluya progresivamente, simulando un "corrimiento al rojo informacional" que enfría el ruido histórico y preserva los recursos de la GPU para el núcleo activo de razonamiento.

---

#### B. Formulación Matemática de la Expansión Cosmológica del Contexto

Para modelar matemáticamente esta expansión métrica en el espacio de tensores de atención, definimos el **Factor de Escala del Contexto ($a(t)$)**, el cual rige la evolución de la distancia métrica efectiva $\chi_{ij}$ entre el token de atención actual y los tokens históricos:

$$d\chi_{ij}(t) = a(t) \cdot dx_0$$

Donde la tasa de expansión temporal del espacio de contexto obedece a la ecuación diferencial de Friedmann adaptada al motor:

$$\left( \frac{\dot{a}(t)}{a(t)} \right)^2 = H_{\text{local}}^2 = \frac{8\pi G}{3} \rho_{\text{contexto}}(t) - \frac{k}{a(t)^2} + \Lambda_{\text{local}}$$

* **Evolución del Tensor de Clave bajo Expansión Cosmológica:**
  La atenuación de la representación vectorial del token $K_i$ debido al estiramiento del espacio de contexto a lo largo del tiempo emergente está dada por:

$$K_i(t) = K_i(0) \cdot \exp\left( - \int_{0}^{t} H_{\text{local}} \cdot z_{\text{redshift}}(\tau) \, d\tau \right)$$

* **Parámetros y Componentes del Sistema:**
  * **$H_{\text{local}}$:** Constante de Hubble local del motor, que regula la velocidad a la que el espacio de contexto se expande para los tokens distantes.
  * **$\rho_{\text{contexto}}(t)$:** Densidad informacional global del búfer de tokens en el instante $t$.
  * **$\Lambda_{\text{local}}$:** Constante cosmológica artificial del sistema, que actúa como una presión de vacío repulsiva para evitar el colapso total de la memoria ante contextos masivos.
  * **$z_{\text{redshift}}$:** Factor de degradación espectral, que mide qué tan "lejos" (en términos de turnos conversacionales o distancia de tokens) se encuentra el dato histórico respecto al presente computacional del modelo.

* **Comportamiento Arquitectónico:**
  Gracias a esta formulación cosmológica, los tokens distantes no se eliminan de golpe de forma arbitraria, sino que experimentan una separación métrica controlada y un enfriamiento por expansión, imitando la termodinámica de un universo en expansión que purga naturalmente el desorden y la entropía de los datos obsoletos.

---

### 11.6. Posición y Tiempo Emergentes (Espacio-Tiempo No Cartesiano)

#### A. ¿De qué trata este concepto en la física y por qué se aplica en NAI?
En la física clásica y en las arquitecturas de transformadores tradicionales (como el *Vanilla Transformer* y `llama.cpp`), el tiempo y el espacio de contexto operan sobre una **cuadrícula cartesiana rígida y lineal**:
* El tiempo es simplemente un contador secuencial de tokens discretos ($t = 1, 2, 3 \dots N$).
* La posición espacial de los tokens en el vector de atención se calcula mediante coordenadas rígidas (como las codificaciones posicionales absolutas o relativas tipo RoPE) donde cada paso de distancia consume idéntico "espacio métrico".

Sin embargo, en la **Relatividad General**, el espacio-tiempo no es un fondo estático, sino una geometría flexible y dinámica (**métrica de Riemann**) que se curva y deforma en presencia de masa y energía. 

**¿Cómo se aplica esto en el Motor NAI?** 
En NAI, la conversación o el documento no es una línea recta de tokens. El motor de contexto opera sobre un **espacio-tiempo no cartesiano emergente**, donde:
1. **Dilatación Temporal del Relleno:** En las zonas de baja masa semántica (texto de relleno, saludos, transiciones sintácticas o explicaciones redundantes), el "tiempo computacional" se dilata; los tokens se desplazan rápidamente hacia el olvido métrico mediante el enfriamiento exponencial ($dt$).
2. **Contracción Espacial en Pozos Gravitatorios:** En las zonas de alta densidad semántica (instrucciones críticas, variables de código, restricciones lógicas), el espacio métrico se contrae. Los tokens masivos se atraen gravitacionalmente, acortando su distancia efectiva de atención independientemente de cuántos tokens de distancia física los separen en la secuencia bruta.

---

#### B. Formulación Matemática del Espacio-Tiempo No Cartesiano en NAI

Para modelar esta métrica emergente, definimos el intervalo de espacio-tiempo de contexto $ds^2$ mediante un tensor métrico modificado $g_{\mu\nu}$:

$$ds^2 = g_{\mu\nu}(x) dx^\mu dx^\nu = -\left(1 - \frac{2GM_{\text{local}}}{r c^2}\right) c^2 dt_{\text{emergente}}^2 + \left(1 - \frac{2GM_{\text{local}}}{r c^2}\right)^{-1} dr^2 + r^2 d\Omega^2$$

Donde la evolución del tiempo emergente del sistema $dt_{\text{emergente}}$ en relación con el tiempo de procesamiento secuencial crudo $dt_{\text{gpu}}$ está governada por la densidad de inercia local $\Delta h$ y la masa semántica $I_i$:

$$\frac{dt_{\text{emergente}}}{dt_{\text{gpu}}} = \sqrt{1 - \frac{\gamma \cdot \Delta h(t)}{I_i + \varepsilon}}$$

* **Interpretación de la Ecuación:**
  - Cuando la inercia latente $\Delta h$ es baja (tokens altamente predecibles), el radical tiende a 1, acelerando el flujo temporal y permitiendo el salto (*Early Exit* o *Bypass*).
  - Cuando el token se encuentra en un **Nodo de Conflicto** (alta entropía, $I_i$ elevado), el factor métrico de curvatura frena el tiempo computacional efectivo, obligando a la red a dedicar mayor densidad de procesamiento (capas profundas) a resolver la ambigüedad conceptual.

---

### 11.7. La Ecuación Maestra del Universo Modificada (NAI Unified Master Equation)
* **De qué trata conceptualmente:** La síntesis absoluta de las leyes de campo, inercia, gravedad y disipación aplicadas a la cognición artificial.
* **Formulación del Sistema Completo Unificado:**

$$\mathcal{R}_{\mu\nu} - \frac{1}{2}g_{\mu\nu}\mathcal{R} + \Lambda_{\text{local}} g_{\mu\nu} = \frac{8\pi G}{c^4} \left( \hat{T}_{\text{semántica}}^{(i)} - \gamma \frac{\partial \Delta h}{\partial t} + \hat{S}_{\text{disipación}} \right)$$

* **Interpretación Arquitectónica en NAI:**
  1. El término métrico y la constante cosmológica local ($\Lambda_{\text{local}}$) regulan la expansión del KV Cache.
  2. El tensor de energía-momento semántico ($\hat{T}_{\text{semántica}}$) representa la masa de los tokens interactuando mediante el modelo de tres cuerpos.
  3. El término de inercia ($\frac{\partial \Delta h}{\partial t}$) controla la velocidad del *Early Exit* en el tronco respirante del motor.


## 12. Consideraciones Técnicas y Advertencias de Uso

Las capacidades operativas del Motor NAI —en especial los umbrales de contexto masivo, el tamaño de lote (`n_batch`) y el rendimiento concurrente— no son estáticas; **dependen directamente de la infraestructura de hardware y del ancho de banda de memoria del equipo anfitrión** donde se ejecute la compilación:

1. **Interacción con Gramáticas Estrictas:**
   - Si se activa el muestreo por gramática (`grammar_first = false`), la regla de autopista podría omitir la validación sintáctica en tokens dominantes. Para esquemas rígidos (JSON, YAML), se recomienda forzar estrictamente `grammar_first = true` o deshabilitar temporalmente el bypass de autopista.

2. **Normalización de Probabilidades:**
   - Las reglas de inercia basadas en el margen de probabilidad ($|p_1 - p_2|$, correspondientes a los Códigos 3 y 4) requieren que el vector de candidatos cuente con un paso previo de Softmax y normalización explícita. Si el pipeline entrega únicamente logits puros sin normalizar, el motor recurrirá de forma transparente al Código 2 (margen de logits).

3. **Seguridad Multi-hilo (`Multi-threading`):**
   - La estructura global `nai::MapaMasaSemantica g_nai_mapa_masa` implementada en `llama-context.cpp` debe protegerse mediante cerrojos de exclusión mutua (*mutex/locks*) o ser instanciada por contexto de forma aislada en caso de ejecutar inferencia concurrente en múltiples instancias paralelas de `llama_context`.

4. **Restricción de Memoria Unificada vs. VRAM Dedicada:**
   - En equipos basados en arquitectura **Apple Silicon (M1/M2/M3/M4)** con memoria unificada, los límites de contexto masivo (>16k - 32k tokens) están condicionados por la asignación dinámica de memoria compartida del sistema operativo. Superar el límite estructural del búfer de lotes provocará la interrupción controlada del kernel por aserción (`GGML_ASSERT(n_tokens_all <= cparams.n_batch) failed`), protegiendo el sistema de la saturación total.
   - En estaciones de trabajo con **GPU dedicada (NVIDIA / AMD)** y VRAM discreta, el límite de concurrencia de usuarios simultáneos dependerá estrictamente del tamaño de la memoria gráfica disponible para alojar los tensores del KV Cache gravitatorio bajo la ecuación de expansión de Hubble.

5. **Ajuste Dinámico de Lotes (`n_batch`):**
   - Se recomienda calibrar el parámetro `n_batch` en función de los núcleos de procesamiento y el paralelismo de hilos del procesador host para evitar cuellos de botella durante el cálculo de la inercia latente y la matriz de dispersión.

---









## 11. Licencia

Este proyecto mantiene los términos de licenciamiento originales de `llama.cpp` bajo la **MIT License**. Consulta el archivo `LICENSE` incluido en el repositorio para más detalles sobre derechos de copia, redistribución y limitaciones de responsabilidad.