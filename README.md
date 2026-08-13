# Proyecto Llama.cpp Modificado: Motor NAI (Inercia Latente, Triaje Quirúrgico y Autopista)

Este repositorio contiene una versión optimizada y extendida de llama.cpp diseñada para reducir la latencia de inferencia y mejorar el procesamiento semántico mediante técnicas de filtrado previo, evaluación elástica de inercia latente y bypass de samplers.

---

## Arquitectura General del Sistema

El pipeline procesa las peticiones en tres capas de optimización:

1. TriageEngine (Filtro Quirúrgico / Prefiltro Header-Only):
   - Intercepta el prompt antes de la tokenización profunda.
   - Clasifica la intención en:
     - DIRECTO: Inferencia estándar/rápida.
     - INDAGACION_CORTA: Flujo asistido con repreguntas breves.
     - PROPUESTA_HIPOTESIS: Generación estructurada de hipótesis de trabajo.
   - Incluye limpieza UTF-8/ASCII y filtrado por coincidencia exacta de palabras clave para evitar falsos positivos.

2. Mapa de Masa Semántica (llama-context.cpp):
   - Mantiene la estructura nai::MapaMasaSemantica g_nai_mapa_masa para el seguimiento del estado latente del contexto.
   - Valida la integridad de buffers (logits, probs, candidates, sampled) mediante aserciones de seguridad y reordenamiento eficiente.

3. Autopista de Inercia y Muestreo Optimizado (sampling.cpp / sampling.h):
   - Evalúa la diferencia de potencia semántica entre los tokens candidatas (DecisionInercia).
   - Aplica un bypass de autopista (vía rápida) saltándose samplers pesados (DRY, XTC, Mirostat, Penalties) cuando existe un token dominantemente claro.

---

## Detalle de Componentes Modificados

### 1. Motor de Triaje (TriageEngine)
Procesa el texto plano para redirigir el flujo sin consumir ciclos del modelo.

- Filtrado de Normalización: Limpieza de signos de puntuación y diacríticos en UTF-8.
- Evaluador Semántico: Clasificación de intención mediante listas de control exactas.

Ejemplo de integración en el flujo principal:
  TriageEngine triage;
  TriageResult res = triage.evaluar(prompt_usuario);

  if (res.flujo == FlujoTipo::INDAGACION_CORTA) {
      // Aplicar plantilla de indagación
  } else if (res.flujo == FlujoTipo::PROPUESTA_HIPOTESIS) {
      // Activar sesgo de razonamiento extendido
  }

### 2. Muestreo y Bypass de Autopista (sampling.cpp / sampling.h)

El módulo de muestreo implementa la estructura DecisionInercia y la función evaluar_inercia_y_equivalencia para determinar si un paso de muestreo requiere la cadena completa de samplers o si puede resolverse de forma directa.

Criterios de Evaluación de Inercia:
- Código 1: Solo existe 1 candidato disponible en el vocabulario filtrado.
- Código 2: El margen de logits entre el primer y segundo candidato supera delta_logit (predeterminado: 2.2f).
- Código 3: Margen de probabilidad |p1 - p2| < epsilon_p (predeterminado: 0.02f) en casos de equivalencia semántica directa.
- Código 4: El token dominante p1 supera por más de 1.3 veces la suma acumulada de la cola (tokens 2 al 5).
- Código 0 (Falla Autopista): Se ejecuta la cadena estándar de samplers (DRY, Top-K, Top-P, XTC, Temperature, Penalties).

Lógica de desvío en common_sampler_sample:
  DecisionInercia dec = evaluar_inercia_y_equivalencia(gsm->cur_p);

  if (dec.es_autopista) {
      LOG_DBG("%s: [Inercia Autopista] Codigo: %d, Margen: %.4f -> Token direct: %d\n",
              __func__, dec.codigo_razon, dec.margen, gsm->cur_p.data[0].id);
      return gsm->cur_p.data[0].id;
  }

### 3. Integración en Contexto (llama-context.cpp)
- Registra dinámicamente las variaciones de tokens en el rastreador de masa semántica nai::MapaMasaSemantica g_nai_mapa_masa.
- Incorpora comprobaciones estrictas de seguridad mediante GGML_ASSERT para la lectura de buffers de logits, probabilidades y candidatos remapeados.

---

## Compilación y Requisitos

### Requisitos Técnicos
- Compilador C++17 o superior (GCC, Clang o MSVC).
- Soporte nativo habilitado para arquitectura ARM64 / Apple Silicon (M1/M2/M3/M4) e x86_64.
- CMake 3.14+.

### Instrucciones de Compilación (Ejemplo con CMake)

  # Clonar/Acceder al proyecto
  cd llama.cpp-nai

  # Crear directorio de compilación
  mkdir build && cd build

  # Configurar proyecto
  cmake .. -DLLAMA_NATIVE=ON

  # Compilar binaries
  cmake --build . --config Release -j

---

## Rendimiento Esperado

1. Reducción de Latencia por Token: Entre un 15% y 35% de aceleración en generación continua gracias al bypass de autopista en tokens con alta certidumbre.
2. Eficiencia en CPU/GPU: Ahorro de ciclos al evitar la ordenación pesada y la evaluación iterativa de la cadena de samplers en más del 40% de los tokens de un flujo conversacional típico.

---

## Consideraciones de Uso y Advertencias Técnicas

1. Interacción con Gramáticas Estrictas:
   - Si se activa el muestreo por gramática (grammar_first = false), la regla de autopista podría saltarse la validación sintáctica en tokens dominantes. Si tu flujo depende de un esquema rígido (JSON, YAML), se recomienda forzar grammar_first = true o desactivar el bypass de autopista.

2. Normalización de Probabilidades:
   - Las reglas de inercia basadas en el margen de probabilidad |p1 - p2| (Códigos 3 y 4) requieren que los candidatos cuenten con un paso previo de Softmax/Normalización. Si el pipeline entrega únicamente logits puros, la decisión recaerá de forma transparente en la regla de diferencia de logits (Código 2).

3. Compatibilidad Multi-hilo:
   - La estructura g_nai_mapa_masa en llama-context.cpp debe protegerse mediante locks o ser instanciada por contexto si se ejecuta inferencia concurrente en múltiples instancias de llama_context.

---

## Puntos de Extensión Futura

- Ajuste Dinámico de Delta Logit: Permite escalar delta_logit dinámicamente según la temperatura o la entropía instantánea de la capa de salida.
- Sincronización con Reasoning Budget: Extensión para aplicar la autopista de forma condicional cuando el modelo está dentro del bloque de pensamiento de cadena de razonamiento (<think>...</think>).

---

## Licencia

Este proyecto mantiene la licencia original de llama.cpp (MIT License). Consulta el archivo LICENSE para más detalles sobre derechos de copia y redistribución.
