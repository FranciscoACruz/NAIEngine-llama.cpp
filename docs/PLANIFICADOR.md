# Planificador NAI: el flujo entrelazado

**Estado:** diseñado y validado en simulación (`tools/nai/sim/`). Todavía **no** está implementado en `llama-server`. Este documento es el puente entre [VISION.md](../VISION.md) y el código: cada fórmula tiene aquí una variable con unidad, un orden de evaluación y un valor calibrado.

---

## 1. El problema

`llama-server` reparte la memoria KV entre las conversaciones de dos formas:

| Modo | Qué pasa bajo carga mixta |
|---|---|
| **KV fijo por slot** (por defecto) | Cada slot recibe C/S tokens. Las conversaciones largas no caben: su contexto se trunca o se cortan. Con menos slots no se trunca, pero las peticiones esperan en fila. |
| **KV unificado** (`--kv-unified`) | Todos comparten la memoria, pero nadie controla el crecimiento. Cuando se llena, la petición falla con `Context size has been exceeded.` |

El segundo caso se observó en el servidor real con `--nai-traza` (18 desbordes en 30 peticiones con el modelo de prueba). El propio código de llama.cpp deja la solución pendiente en `server-context.cpp`:

```cpp
// TODO: try to terminate only the largest active slot/sequence and continue with the rest
```

El planificador NAI resuelve esto **entrelazando** las fórmulas de VISION.md sobre unas pocas variables compartidas.

---

## 2. Variables compartidas (unidades comunes)

| Símbolo | Significado | Unidad |
|---|---|---|
| C | capacidad del KV | tokens |
| M | masa de una tarea = prompt + margen · salida estimada | tokens |
| r | memoria libre comprometida = C − Σ masa comprometida de las tareas activas | tokens |
| ρ | ocupación real = KV usado / C | 0..1 |
| t | espera de una tarea en cola | pasos del servidor |
| q | masa real / masa estimada de una tarea terminada | adimensional |
| m | margen de masa (lo corrige el campo) | adimensional |
| Λ | reserva para tareas cortas | fracción de C |

La **masa comprometida** de una tarea activa es `max(KV que ya ocupa, prompt + m · salida_estimada)`. Así, una tarea que rebasa su estimación compromete exactamente lo que ya ocupa.

---

## 3. Ecuaciones y orden de evaluación (cada paso del servidor)

```
1. Llegadas → cola
2. Matriz S + tiempo emergente (prioridad, menor = primero):
       P(x) = costo_restante(x) / (1 + t(x)/τ)
       costo_restante = prompt pendiente + max(1, m·salida_est − generados)
3. Para cada tarea en orden de P, mientras haya slots libres:
   a. Carril rápido: una tarea compleja no toma los últimos S_r slots libres
   b. Schwarzschild (horizonte): admitir solo si
          r_s(x) ≤ C − Σ comprometida − Λ·C·[x es compleja]
          con r_s(x) = max(KV de x, prompt + m·salida_est)
      (con el servidor vacío siempre se admite: nunca hay inanición total)
   c. N cuerpos (slot): argmax sobre slots libres de
          F = r_s · (1 + caché_slot) / d²,   d = 1 − prefijo_compartido/prompt + 0.05
   d. Si la tarea estaba hibernada: restaurar su KV desde disco
4. Lote: prefill de las tareas con menos prompt pendiente primero (n_batch), + 1 token por tarea generando
5. Si el KV real se desborda: hibernar la tarea de mayor P (guardar KV a disco, vuelve a la cola)
   — nunca fallar la petición
6. Tareas terminadas: liberar slot (conservar prefijo en caché) y aplicar el campo:
       log_q̄ ← log_q̄ + κ · (log q − log_q̄)
       m      = clamp(m₀ · exp(log_q̄), 0.5, 4)
7. t ← t + 1 para las tareas en cola
```

### Cómo se mapea cada fórmula de VISION.md

| Fórmula | Papel | Resultado en simulación |
|---|---|---|
| **Matriz S / restitución** | Prioridad por costo restante: lo rápido sale primero y libera el carril | Esencial: latencia p95 de cortas de 328 s a 4.8 s (contra KV fijo de 4 slots) |
| **Tiempo emergente** | Envejecimiento (τ = 400 pasos): la espera dilata la prioridad | Esencial: sin él, las complejas pueden esperar indefinidamente |
| **Schwarzschild** | Horizonte de admisión con masa comprometida | **El mecanismo principal:** elimina los desbordes cuando las estimaciones son razonables |
| **Reserva Λ y carril rápido** | "Dejar espacio entre tareas pequeñas y complejas" | Protegen a las cortas cuando solo llegan complejas |
| **N cuerpos** | Asignación de slot por masa y prefijo compartido | Útil, pero llama.cpp ya hace afinidad por prefijo (LCP); la ganancia adicional es pequeña |
| **Ecuación de campo** | **Masa adaptativa:** m se corrige con la masa observada. Es el amortiguamiento real del sistema (EMA = inercia) | Esencial cuando el estimador tiene sesgo: m converge solo a ≈ 3 cuando la salida real es 3× la estimada |
| **Friedmann (anticipación por flujo)** | Proyectar ρ con el flujo de masa no comprometida y actuar antes del desborde | **No aportó** en ningún escenario y cuesta 1–4 %. Se deja disponible y apagada por defecto |
| **Hibernación** | Expulsar guardando el KV (mecanismo de las sesiones NAI) en lugar de fallar | Red de seguridad: en los escenarios probados no se activó, porque el horizonte y el campo evitaron los desbordes |

### Lo que la simulación cambió del diseño original

1. **Limitar el ritmo de admisión (Friedmann como regulador de entrada) no evita desbordes.** Los causan las tareas que **ya corren** y generan más de lo estimado. Por eso el control actúa sobre la estimación de masa (campo) y sobre la masa en movimiento (hibernación).
2. **El amortiguamiento sí es un flujo de masa, pero medido por tarea, no por paso.** La relación masa real / estimada, suavizada al terminar cada tarea, estabiliza el sistema. El flujo instantáneo de tokens es demasiado ruidoso: el prefill aparece como una "amenaza" aunque ya esté comprometido.
3. **La anticipación requiere histéresis.** Si el umbral de admisión y el de hibernación son el mismo, el sistema hiberna y readmite en ciclo. Se resolvió con una banda: se hiberna si la proyección > 1.0 y se readmite solo si queda bajo 1 − Λ − 0.05.

---

## 4. Resultados de la simulación

16 slots, 32 768 tokens de KV, 600 peticiones (30 % complejas), 3 semillas. El modelo de tiempo por paso es sintético y se calibra con trazas reales (sección 6).

| Escenario | Política | Íntegras¹ | p95 cortas | p95 complejas | tok/s | Fallidas | Truncadas |
|---|---|---|---|---|---|---|---|
| mixto | llama.cpp KV fijo, 16 slots | 70 % | 4.4 s | 26 s | 499 | 0 | 182 |
| mixto | llama.cpp KV fijo, 4 slots | 99 % | 328 s | 344 s | 251 | 0 | 5 |
| mixto | llama.cpp KV unificado | 84 % | 3.4 s | 25 s | 373 | 94 | 0 |
| mixto | **NAI completo** | **100 %** | 4.8 s | 115 s | 399 | 0 | 0 |
| subestimación² | NAI sin control | 98 % | 4.6 s | 82 s | 420 | 10 | 0 |
| subestimación² | **NAI completo** | **100 %** | 4.5 s | 114 s | 399 | 0 | 0 |
| extremo³ | llama.cpp KV unificado | 62 % | 4.5 s | 31 s | 348 | 226 | 0 |
| extremo³ | NAI sin control | 99 % | 7.9 s | 208 s | 390 | 8 | 0 |
| extremo³ | **NAI completo** | **100 %** | 8.5 s | 279 s | 336 | 0 | 0 |

¹ Peticiones completas sin error y sin contexto truncado. Las latencias de las políticas con fallos o truncación solo cuentan las peticiones que sí terminaron, por eso se ven bajas.
² El estimador predice 1/3 de la salida real.
³ Ráfagas, subestimación, error de estimación grande y salidas de cola pesada (Pareto).

**Lectura honesta:**
- **NAI completo es la única política con 100 % de integridad en los cinco escenarios.**
- **Contra la línea base sin pérdidas** (KV fijo de 4 slots), el throughput sube +59 % y la latencia p95 de las cortas baja de 328 s a 4.8 s.
- **Las complejas esperan más** que en las políticas que truncan o fallan, porque NAI no sacrifica a ninguna. Ese es el costo de priorizar lo rápido.
- **En el escenario extremo**, NAI completo rinde 14 % menos que la versión sin control, pero esa versión hace fallar 8 peticiones.

Gráficas: [`docs/nai-sim/resumen.png`](nai-sim/resumen.png), [`serie_subestimacion.png`](nai-sim/serie_subestimacion.png) y [`barrido_H_alfa.png`](nai-sim/barrido_H_alfa.png).

### Parámetros recomendados

| Parámetro | Valor | Nota |
|---|---|---|
| τ (envejecimiento) | 400 pasos | ≈ 15–20 s con modelos de 7B |
| Λ (reserva) | 0.03 | Con el campo activo no hace falta más |
| m₀ (margen inicial) | 1.0 | El campo lo ajusta solo |
| κ (ganancia del campo) | 0.05 por tarea | Converge en unas 50 tareas |
| S_r (carril rápido) | 2 slots de 16 | Reservados para tareas cortas |
| Anticipación por flujo | Apagada | H = 40, α = 0.1 y banda 0.05 si se activa |

---

## 5. Plan de implementación en `llama-server`

Requiere `--kv-unified`: la memoria compartida es la condición del problema.

1. **Estimador de masa.** Usar `n_predict`/`max_tokens` si viene en la petición. Si no, la media móvil de la salida por sesión, y si no hay historial, una constante por clase del triaje. Todo multiplicado por el margen m del campo.
2. **Cola con prioridad.** Reemplazar el FIFO de `queue_tasks` (`server-queue.cpp`) por un orden según P(x), con `t_llegada_us` (ya agregado para la traza) como base del envejecimiento.
3. **Horizonte.** En `get_available_slot`/`launch_slot_with_task`, rechazar temporalmente (diferir) si `r_s > C − Σ comprometida − Λ·C`.
4. **Hibernación.** Cuando `llama_decode` devuelve 1 (KV lleno), en lugar de `send_error`, guardar el slot de mayor P con `llama_state_seq_save_file` (el mismo camino que `--nai-sesiones`), liberar su secuencia, re-encolarlo y reintentar el lote. Es el TODO del propio llama.cpp.
5. **Campo.** Al liberar cada slot, actualizar `log_q̄` con `n_decoded` frente a la salida estimada.
6. **Validación.** Comparar con `nai_bench.py --carga-mixta` antes y después, con las mismas trazas.

---

## 6. Calibración con datos reales

```console
# 1) carga mixta real con traza
python3 tools/nai/nai_bench.py -m modelo.gguf --carga-mixta 200 --slots 16 --ctx 32768 \
        --flags-servidor="--kv-unified"

# 2) ajustar el modelo de tiempo y repetir la carga en el simulador
python3 tools/nai/sim/nai_planificador_sim.py --traza nai-bench-logs/traza-tareas.csv \
        --calibrar nai-bench-logs/traza-pasos.csv --capacidad 32768 --slots 16
```

El simulador imprime primero la latencia **real** medida en la traza y después la de cada política simulada. Si la política de llama.cpp simulada no se parece a la real, hay que ajustar el modelo antes de creer en las demás.
