# Ruta de la información en NAI, modelo por modelo

Este documento sigue el camino de la información desde que entra a NAI hasta que se vuelve parte del modelo, e indica qué modelo interviene en cada paso. Distingue lo que **ya existe** de lo que está **planeado** (README §9.5).

**Hoy un solo modelo hace todo**: conversa y, en un contexto aparte de 1,024 tokens (el *motor de memoria*), guarda, clasifica y vigila lo que se aprende. El plan es repartir ese trabajo entre modelos especializados, como las partes de un cerebro, sin cambiar las rutas: cambia quién hace cada paso, no el camino.

## 1. Mapa general

```mermaid
flowchart LR
    F["Fuentes<br/>archivos .md, URL, PDF,<br/>nai-fuentes.txt, conversación"] --> G["Guardián<br/>naturaleza"]
    G -->|manipulación en documento| X["Descartado"]
    G --> R["Región<br/>dada o clasificada"]
    R --> M[("Memoria NAI<br/>biblioteca.mem<br/>sesiones .mem")]
    M -->|solo si la pregunta la toca| P["Modelo principal<br/>conversación"]
    P --> U["Respuesta"]
    P -->|turno, en segundo plano| G
    M -->|exportar JSONL| E["Reentrenamiento<br/>nai_reentrenar.py"]
    I["Carta de identidad<br/>docs/IDENTIDAD_NAI.md"] --> E
    E -->|adaptador LoRA| P
```

## 2. Los modelos y su papel

| Modelo | Parte del cerebro | Qué recibe | Qué produce | Estado |
|---|---|---|---|---|
| **Modelo principal** (1.5–3B instruct) | Corteza: conversar y razonar | Mensaje, recuerdos relevantes, últimos 3 turnos | La respuesta | **Existe** |
| **Motor de memoria** (hoy: el mismo modelo, contexto aparte) | Hipocampo que guarda | Cada trozo nuevo | Región, naturaleza, claves, frase IDEA al olvidar | **Existe** dentro del principal |
| **Guardián** (dentro del motor) | Filtro de lo que se aprende | Cada trozo nuevo | Naturaleza: conocimiento, sabiduría, desahogo, manipulación | **Existe**: patrones fijos + predicción |
| **Identidad LoRA** (sobre el principal) | Corteza prefrontal: carácter, ética, empatía | Carta de identidad + biblioteca | Un adaptador que se carga con `--lora` | **Existe** el proceso; falta entrenarlo |
| **Cerebelo** (0.5B instruct) | Automático y predictivo | Trozos nuevos; borradores de la respuesta | El trabajo del motor de memoria; tokens propuestos al principal (especulación); a qué especialista va cada pregunta | **Planeado** |
| **Hipocampo de embeddings** (modelo de embeddings chico) | Asociación por significado | Cada trozo y cada pregunta | Un vector de significado: «mascota» encuentra «perro» | **Planeado** |
| **Especialistas** (código, matemáticas; 1.5B) | Áreas especializadas | Preguntas que el cerebelo les envía | Respuestas de su campo | **Planeado**, cargados bajo demanda |
| **Memoria NAI** (no es un modelo) | Cuerpo calloso: lo que todos comparten | Lo que guardan todos | Recuerdos para todos | **Existe** |

## 2b. Modelos concretos y conocimientos de cada uno

Hoy NAI corre con **un solo modelo** (el que pasas con `-m`; en las pruebas, Qwen2.5-0.5B-Instruct). La tabla dice qué modelo se propone para cada papel cuando se reparta el trabajo. Todos son de la familia Qwen para compartir vocabulario (regla 1 y 2, §6), salvo el oído y la voz, que no tocan la memoria.

| Papel | Modelo propuesto (GGUF) | Tamaño q4 aprox. | Conocimientos y tareas que le corresponden |
|---|---|---|---|
| **Principal** (conversación) | Qwen3-4B; con 8 GB justos, Qwen3-1.7B | ~2.5 GB / ~1.1 GB | Las regiones de la vida y del sentido: conciencia, ética, espiritualidad, empatía, psique, filosofía, arte, medicina tradicional, biología, agricultura, supervivencia, finanzas personales, cocina, compras, deportes, trámites, entretenimiento, general. Lleva el LoRA de identidad |
| **Razonamiento** (lógica y cálculo) | El mismo principal en modo razonamiento (Qwen3 lo trae); no ocupa memoria extra | 0 | Lógica, matemáticas, física, química, mecánica, aerodinámica y la parte de cálculo de ingeniería |
| **Especialista en código** (bajo demanda) | Qwen2.5-Coder-1.5B-Instruct | ~1.0 GB | Ciberseguridad técnica (código, configuraciones, análisis), programación e ingeniería de software |
| **Cerebelo** (memoria y reparto) | Qwen3-0.6B | ~0.4 GB | Ningún tema propio: guarda, clasifica región y naturaleza, propone tokens al principal (especulación) y decide a quién va cada pregunta |
| **Hipocampo** (asociación) | Qwen3-Embedding-0.6B (Apache 2.0) | ~0.4 GB | Ningún tema propio: convierte cada recuerdo y cada pregunta en un vector de significado para relacionarlos |
| **Oído** (voz a texto) | Whisper large-v3-turbo con whisper.cpp; con 8 GB justos, Whisper small | ~0.6 GB / ~0.2 GB | Transcribir lo que hablas, en español |
| **Voz** (texto a voz) | La voz del sistema (`say` en macOS) para empezar; después OuteTTS 1.0 0.6B | 0 / ~0.4 GB | Leer en voz alta la respuesta de NAI |

Notas:

- **La memoria la comparten todos.** Una región dice de qué trata un recuerdo, no qué modelo lo guarda: cuando el especialista en código responde, también ve los recuerdos de ciberseguridad que aprendió NAI.
- **El principal hace también el razonamiento.** Un modelo de matemáticas aparte costaría más de 1 GB; Qwen3 ya trae un modo de razonamiento paso a paso, así que basta con activarlo para esas regiones.
- **El hipocampo con licencia libre.** jina-embeddings-v5-text-small mide algo mejor en las pruebas multilingües, pero su licencia es no comercial (CC-BY-NC 4.0); Qwen3-Embedding-0.6B es Apache 2.0 y sirve también si NAI llega a AmigoCash.
- **Los nombres y tamaños son la propuesta de noviembre de 2026**: se confirman al implementar cada pieza, midiendo precisión, velocidad y memoria en la Mac.

## 2c. Oído y voz

```mermaid
flowchart LR
    MIC["Micrófono"] --> O["Oído<br/>Whisper, whisper.cpp"]
    O -->|texto| S["Servidor NAI<br/>memoria + principal"]
    S -->|respuesta en texto| V["Voz<br/>say / OuteTTS"]
    V --> BOC["Bocina"]
```

La voz entra y sale **por fuera** del servidor: el oído convierte el audio en texto antes de entrar y la voz lee la respuesta al salir. La memoria, el guardián y la identidad no cambian. Lo que se guarda es el texto transcrito, no el audio.

| Pieza | Qué existe ya | Qué falta |
|---|---|---|
| Oído | whisper.cpp es del mismo ecosistema ggml y corre en Metal en Apple Silicon | Un script o cliente que grabe, transcriba y envíe a `/v1/chat/completions` |
| Voz | `say` viene con macOS y tiene voces en español; el fork trae `llama-tts` para modelos OuteTTS | Leer la respuesta al llegar. OuteTTS 1.0 requiere una versión de `llama-tts` con soporte para su formato nuevo |

## 3. Ruta de aprendizaje (entrada)

```mermaid
flowchart TD
    A["Fuente"] --> B["Lector<br/>common/nai-lector.cpp<br/>texto, HTML, PDF, Word"]
    B --> C["Trocear<br/>párrafos de hasta 600 caracteres"]
    C --> D{"Guardián<br/>¿manipulación?<br/>patrones fijos, sin modelo"}
    D -->|sí, documento| X["Descartado<br/>guardian_rechazados +1"]
    D -->|no| E["Naturaleza por predicción<br/>conocimiento · sabiduría · desahogo"]
    E --> F["Región<br/>la del archivo o por predicción"]
    F --> G["Codificar<br/>diccionario de tokens, portable"]
    G --> H["Claves<br/>palabras más sorprendentes"]
    H --> I[("biblioteca.mem<br/>o sesión .mem")]
    I --> J{"¿Cabe en el presupuesto?"}
    J -->|no| K["Bajar de nivel lo menos importante<br/>exacto → idea → pista → olvido"]
```

| Paso | Lo hace hoy | Lo hará con el plan |
|---|---|---|
| Leer y trocear | Código, sin modelo | Igual |
| Detectar manipulación | Patrones fijos, sin modelo | Igual (determinista a propósito) |
| Naturaleza (las otras tres) | Motor de memoria, por predicción | Cerebelo |
| Región | La del archivo; si no, motor por predicción | La del archivo; si no, cerebelo |
| Codificar | Diccionario de tokens del modelo que guarda | Igual, con el vocabulario del cerebelo |
| Claves | Sorpresa medida por el motor | Sorpresa del cerebelo + vector del hipocampo |
| Frase IDEA al olvidar | Motor de memoria (`generar`) | Cerebelo |

**Importancia al olvidar** = factor de naturaleza × [peso de la región × (1 + 0.5 × usos) + 1 / (1 + días sin usarse)], con factor 1.5 para sabiduría, 1 para conocimiento, 0.5 para desahogo y 0.25 para manipulación (solo existe en conversaciones; en documentos se descarta). Lo menos importante se drena primero.

## 4. Ruta de conversación (consulta)

```mermaid
sequenceDiagram
    participant U as Usuario
    participant S as Servidor
    participant M as Memoria NAI
    participant P as Modelo principal
    U->>S: mensaje
    S->>M: claves del mensaje
    M-->>S: recuerdos que coinciden + 1 asociado opcional<br/>(nada si la pregunta no toca la memoria)
    S->>S: sistema + recuerdos + últimos 3 turnos + mensaje<br/>(se recorta si no cabe, nunca se rechaza)
    S->>P: prompt
    P-->>U: respuesta
    S-)M: en segundo plano: guardar el turno<br/>(guardián, región, codificar)
```

| Paso | Lo hace hoy | Lo hará con el plan |
|---|---|---|
| Buscar recuerdos | Claves (palabras), sin modelo | Claves + vector del hipocampo |
| Elegir quién responde | Siempre el principal | El cerebelo reparte: principal o especialista |
| Generar la respuesta | Modelo principal | Principal con borrador del cerebelo (más rápido) |
| Guardar el turno | Motor, en segundo plano; cede el paso si hay peticiones activas | Cerebelo, en segundo plano |

## 5. Ruta de consolidación (de la memoria a los pesos)

```mermaid
flowchart LR
    A[("biblioteca.mem")] -->|exportar| B["recuerdos.jsonl<br/>con región y naturaleza<br/>sin manipulación"]
    C["docs/IDENTIDAD_NAI.md"] --> D
    B --> D["datos<br/>arma los ejemplos"]
    D --> E["fase 1: identidad ×2, prioritarios, sabiduría"]
    D --> F["fase 2: generales, desahogo con contexto, repaso"]
    E --> G["entrenar<br/>MLX, Apple Silicon"]
    F --> G
    G --> H{"medir<br/>¿algún tema empeoró?"}
    H -->|sí| X["Adaptador rechazado"]
    H -->|no| I["convertir a GGUF"]
    I --> J["llama-server --lora"]
```

| Naturaleza | Cómo entra al entrenamiento |
|---|---|
| Identidad (la carta) | Fase 1, dos pasadas por época. NAI la cita como lo que su autor le enseñó |
| Sabiduría | Fase 1, con los temas prioritarios |
| Conocimiento | Según el peso de su región |
| Desahogo | Fase 2, peso 1, solo con su contexto: «una persona bajo presión, no una verdad sobre la humanidad»; sin preguntas |
| Manipulación | Nunca |

Además, de cada recuerdo sale un ejemplo que enseña a reconocer su naturaleza, así que el guardián mejora con cada reentrenamiento.

Hoy este proceso se corre a mano (`python3 tools/nai/nai_reentrenar.py todo …`, README §3.6). El plan es dispararlo solo cuando la biblioteca acumule suficiente material nuevo.

## 6. Reglas que no cambian al repartir el trabajo

1. **El vocabulario de la memoria es fijo.** Los recuerdos se guardan como IDs de token del modelo que guarda. Si ese modelo cambia de familia (por ejemplo, de Qwen a Llama), los recuerdos dejan de leerse. Con el cerebelo, él será el dueño fijo de la memoria. **Ojo, hoy:** la huella de los archivos `.mem` incluye el tamaño del modelo, así que cambiar el modelo con `-m` (por ejemplo, de 0.5B a 3B) empieza una memoria nueva aunque el vocabulario sea el mismo. Pendiente: que con el diccionario de tokens la huella dependa solo del vocabulario.
2. **La especulación solo funciona dentro de una familia.** El cerebelo y el principal deben compartir vocabulario (por ejemplo, Qwen 0.5B con Qwen 3B).
3. **La manipulación se detecta sin modelo.** Patrones fijos y deterministas: ningún modelo, por grande que sea, decide si se aprende un intento de manipularlo.
4. **La memoria solo se consulta cuando la pregunta la toca.** Lo asociado se ofrece como comentario opcional, nunca como dato impuesto.
5. **Ninguna conversación se bloquea por tamaño.** Si algo no cabe, se recorta; lo recortado ya está en la memoria.

## 7. Memoria en un equipo de 8 GB (estimado)

| Configuración | Modelos cargados | Total aprox. (sin KV) |
|---|---|---|
| **Hoy** | Un modelo (principal + motor de memoria) | 0.4 GB (0.5B) a 2 GB (3B) |
| **Mínima con voz** | Qwen3-1.7B + Qwen3-0.6B + Qwen3-Embedding-0.6B + Whisper small + `say` | ~2.1 GB |
| **Completa** | Qwen3-4B + Qwen3-0.6B + Qwen3-Embedding-0.6B + Whisper turbo + OuteTTS 0.6B | ~4.3 GB; el especialista en código (+1 GB) se carga solo cuando hace falta |

Más el KV de cada modelo. Arranca con `-fitt 2048` para dejar margen al sistema. En 8 GB, la configuración completa queda justa: conviene empezar por la mínima y subir midiendo.
