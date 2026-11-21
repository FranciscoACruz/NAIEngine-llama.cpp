#!/usr/bin/env python3
# [NAI] Simulador del planificador entrelazado (antes de implementarlo en llama-server)
#
# Modela el servidor paso a paso: S slots, una memoria KV compartida de C tokens, prefill por
# trozos de n_batch tokens intercalado con la generación, caché de prefijos por slot y tareas
# cuya salida real difiere de la estimada. Compara la política de llama.cpp contra el flujo
# NAI, donde todas las fórmulas comparten las mismas variables:
#
#   M   masa de la tarea      = tokens de KV que ocupará (prompt + salida estimada)   [tokens]
#   r   memoria libre         = C − KV comprometido por tareas activas                [tokens]
#   ρ   ocupación             = KV realmente usado / C                                [0..1]
#   t   espera                = pasos en cola                                         [pasos]
#   Φ   flujo de masa         = (M_entra − M_sale) / C por paso                       [1/paso]
#
#   Schwarzschild  admitir solo si r_s(M) < r − Λ·C (las tareas cortas sí pueden usar la reserva Λ)
#   Matriz S       prioridad = costo restante estimado (lo rápido sale primero)
#   Tiempo emerg.  la prioridad mejora con la espera: costo / (1 + t/τ)  (evita inanición)
#   N cuerpos      slot = argmax  M_tarea·(1 + M_caché) / d²,  d = 1 − prefijo_compartido/prompt + ε
#   Friedmann      la masa comprometida (horizonte) ya cubre lo esperado; el riesgo es el flujo
#                  de masa NO comprometida Φ̄ₑ (tokens de tareas que rebasaron su estimación,
#                  suavizado = inercia). Si (KV comprometido + Φ̄ₑ·C·H)/C > ρ_max el colapso llega en
#                  H pasos: no se admite y se hiberna la tarea de mayor costo restante ANTES del
#                  desbordamiento (amortiguamiento proporcional a la velocidad, no a la posición).
#                  Histéresis: se readmite solo si la proyección queda bajo 1 − Λ − banda.
#   Campo          masa adaptativa: el margen de estimación se corrige con la masa real observada,
#                  m ← m + κ·(q − m),  q = salida_real / salida_estimada de las tareas terminadas
#   Hibernación    una tarea expulsada guarda su KV a disco (como las sesiones NAI) y lo recupera
#                  al volver, en lugar de perder su progreso
#
# Hallazgo del simulador (ver docs/PLANIFICADOR.md): limitar el RITMO de admisión no evita el
# desbordamiento, porque lo causan las tareas que ya corren y generan más de lo estimado; por
# eso el control actúa sobre la masa en movimiento (anticipación + hibernación) y sobre la
# estimación de masa (campo), no sobre un ritmo de admisión.
#
# Uso:
#   python3 tools/nai/sim/nai_planificador_sim.py --salida informe/     # experimentos + gráficas
#   python3 tools/nai/sim/nai_planificador_sim.py --traza tareas.csv    # repetir una traza real
#
# El modelo de tiempo por paso es sintético; calíbralo con una traza real (--nai-traza en
# llama-server) usando --calibrar pasos.csv.

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------------------------


@dataclass
class Config:
    capacidad: int = 32768          # C: tokens de KV totales
    slots: int = 16                 # S (servidor con muchas conversaciones: limita la memoria)
    n_batch: int = 512              # tokens de prefill por paso (entre todos los slots)
    # tiempo por paso [ms] = base + por_gen·generados + por_pre·prefill + por_kv·KV usado
    t_base: float = 12.0
    t_por_gen: float = 0.30
    t_por_pre: float = 0.05
    t_por_kv: float = 0.0001
    t_swap_por_token: float = 0.03  # ms por token de KV al hibernar o restaurar (≈ 57 KB/token a 2 GB/s)


@dataclass
class Politica:
    nombre: str
    kv_estatico: bool = False       # llama.cpp por defecto: C/S tokens fijos por slot
    slots_usables: int = 0          # 0 = todos; con KV fijo, menos slots = más contexto por slot
    ordenar: str = "fifo"           # fifo | costo
    envejecimiento_tau: float = 0.0 # τ en pasos (0 = sin envejecimiento)
    horizonte: bool = False         # Schwarzschild
    reserva: float = 0.0            # Λ (fracción de C)
    margen_estimacion: float = 1.2  # r_s = prompt + margen·salida_estimada
    afinidad: str = "prefijo"       # prefijo (llama.cpp) | gravedad (N cuerpos)
    prefill_corto_primero: bool = False
    slots_rapidos: int = 0          # slots que una tarea compleja nunca puede ocupar (carril rápido)
    hibernar: bool = False          # expulsar guardando el KV a disco en lugar de perder el progreso
    anticipar: bool = False         # Friedmann: actuar si ρ + Φ̄·H > ρ_max
    H: int = 40                     # pasos de anticipación
    rho_max: float = 1.0            # hibernar si la proyección rebasaría la memoria
    banda: float = 0.05             # histéresis: readmitir solo con la proyección < 1 − Λ − banda
    alfa_flujo: float = 0.1         # inercia del flujo (EMA); menor = más inercia
    adaptar_masa: bool = False      # campo: corregir el margen con la masa observada
    kappa: float = 0.05             # ganancia de la corrección (por tarea terminada)


@dataclass
class Tarea:
    id: int
    llegada: float                  # ms
    familia: int
    prefijo: int                    # tokens del prompt compartidos con su familia
    prompt: int
    salida: int                     # real (desconocida para el planificador)
    salida_est: int                 # estimada
    clase: str
    prefill_hecho: int = 0
    generados: int = 0
    slot: int = -1
    inicio: float = -1.0
    primer_token: float = -1.0
    fin: float = -1.0
    espera_pasos: int = 0
    preempciones: int = 0
    hibernada: bool = False         # su KV está en disco: al volver no repite trabajo
    fallida: bool = False           # llama.cpp respondió "Context size has been exceeded." 
    truncada: bool = False
    descartados: int = 0            # tokens quitados por context shift (KV fijo por slot)

    @property
    def kv(self) -> int:
        return self.prefill_hecho + self.generados - self.descartados

    def comprometido(self, margen: float) -> int:
        return max(self.kv, self.prompt + int(margen * self.salida_est))


@dataclass
class Slot:
    id: int
    tarea: Tarea | None = None
    familia_cache: int = -1
    cache: int = 0                  # tokens de prefijo que conserva mientras está libre


@dataclass
class Resultado:
    politica: str
    tareas: list[Tarea]
    t_total: float
    tokens_generados: int
    preempciones: int
    tokens_perdidos: int
    hibernaciones: int = 0
    serie_t: list[float] = field(default_factory=list)
    serie_rho: list[float] = field(default_factory=list)
    serie_margen: list[float] = field(default_factory=list)
    serie_cola: list[int] = field(default_factory=list)


def generar_carga(n: int, tasa_por_s: float, frac_compleja: float, semilla: int,
                  rafaga: bool = False, sigma_est: float = 0.5, familias: int = 6,
                  sesgo_est: float = 1.0, cola_pesada: bool = False) -> list[Tarea]:
    # cola_pesada: salidas con distribución de Pareto (unas pocas respuestas enormes)
    # sesgo_est < 1: el estimador subestima sistemáticamente la salida (p. ej. 0.5 = la mitad)
    rnd = random.Random(semilla)
    prefijos = [rnd.randint(200, 1200) for _ in range(familias)]
    t = 0.0
    tareas = []
    for i in range(n):
        tasa = tasa_por_s
        if rafaga:  # alterna 20 s de carga x2.5 con 20 s de carga x0.4
            tasa *= 2.5 if int(t / 20000) % 2 == 0 else 0.4
        t += rnd.expovariate(tasa) * 1000.0
        compleja = rnd.random() < frac_compleja
        fam = rnd.randrange(familias)
        if compleja:
            extra = int(rnd.lognormvariate(math.log(2500), 0.5))
            salida = int(rnd.lognormvariate(math.log(700), 0.4))
        else:
            extra = int(rnd.lognormvariate(math.log(150), 0.5))
            salida = int(rnd.lognormvariate(math.log(80), 0.5))
        if cola_pesada and rnd.random() < 0.08:
            salida = int(salida * rnd.paretovariate(1.2))
        salida = max(4, min(salida, 8000))
        prompt = prefijos[fam] + max(8, min(extra, 12000))
        est = max(4, int(sesgo_est * salida * math.exp(rnd.gauss(0.0, sigma_est))))
        tareas.append(Tarea(i, t, fam, prefijos[fam], prompt, salida, est, "compleja" if compleja else "corta"))
    return tareas


def simular(cfg: Config, pol: Politica, carga: list[Tarea], max_pasos: int = 400000) -> Resultado:
    tareas = [Tarea(**{k: v for k, v in vars(t).items()}) for t in carga]  # copia limpia
    pendientes = sorted(tareas, key=lambda x: x.llegada)
    cola: list[Tarea] = []
    n_slots = pol.slots_usables or cfg.slots
    slots = [Slot(i) for i in range(n_slots)]
    cap_slot = cfg.capacidad // n_slots
    C = cfg.capacidad
    t = 0.0
    paso = 0
    flujo = 0.0             # Φ̄: flujo de masa total suavizado (diagnóstico)
    flujo_exc = 0.0         # Φ̄ₑ: flujo de masa no comprometida suavizado (control)
    margen = pol.margen_estimacion
    log_q = 0.0             # campo: log(salida real / estimada) promedio
    swap_ms = 0.0           # tiempo de hibernar/restaurar acumulado en el paso
    res = Resultado(pol.nombre, tareas, 0.0, 0, 0, 0)
    terminadas = 0

    def kv_usado() -> int:
        return sum(s.tarea.kv for s in slots if s.tarea) + sum(s.cache for s in slots if not s.tarea)

    def kv_activo() -> int:
        return sum(s.tarea.kv for s in slots if s.tarea)

    def comprometido() -> int:
        return sum(s.tarea.comprometido(margen) for s in slots if s.tarea)

    def expulsar(victima: Slot) -> int:
        nonlocal swap_ms, terminadas
        x = victima.tarea
        liberado = x.kv
        x.preempciones += 1
        x.slot = -1
        victima.tarea = None
        if pol.hibernar:            # KV a disco: no se pierde progreso, vuelve a la cola
            res.hibernaciones += 1
            swap_ms += cfg.t_swap_por_token * x.kv
            x.hibernada = True
            cola.append(x)
        else:                       # como llama.cpp: la petición falla con error (observado en la traza real)
            res.preempciones += 1
            res.tokens_perdidos += x.kv
            x.fallida = True
            terminadas += 1
        return liberado

    def prioridad(x: Tarea) -> float:
        if pol.ordenar == "fifo":
            return x.llegada
        costo = (x.prompt - x.prefill_hecho) + max(1, x.salida_est - x.generados)
        if pol.envejecimiento_tau > 0:
            costo /= 1.0 + x.espera_pasos / pol.envejecimiento_tau
        return costo

    while terminadas < len(tareas) and paso < max_pasos:
        paso += 1
        while pendientes and pendientes[0].llegada <= t:
            cola.append(pendientes.pop(0))
        if not cola and all(s.tarea is None for s in slots) and pendientes:
            t = pendientes[0].llegada
            continue

        m_sale = 0

        # --- admisión ---
        cola.sort(key=prioridad)
        colapso = pol.anticipar and (comprometido() + flujo_exc * C * pol.H) / C > pol.rho_max
        libres = [s for s in slots if s.tarea is None]
        i = 0
        while libres and i < len(cola):
            x = cola[i]
            r_s = max(x.kv, x.prompt + int(margen * x.salida_est))
            hay_activos = kv_activo() > 0
            if pol.horizonte and hay_activos:   # con el servidor vacío siempre se admite
                reserva = pol.reserva * C if x.clase == "compleja" else 0.0
                if pol.anticipar:               # cuenta la masa no comprometida que viene + histéresis
                    reserva += flujo_exc * C * pol.H + pol.banda * C
                if r_s > C - comprometido() - reserva:
                    i += 1          # no cabe: se queda en cola (sigue envejeciendo)
                    continue
            if colapso and hay_activos:
                break               # Friedmann: se aproxima el colapso, no se admite masa nueva
            if x.clase == "compleja" and len(libres) <= pol.slots_rapidos and hay_activos:
                i += 1              # carril rápido: los últimos slots libres son para tareas cortas
                continue
            # N cuerpos / afinidad de prefijo
            def solapa(s: Slot) -> int:
                return min(s.cache, x.prefijo) if s.familia_cache == x.familia else 0
            if pol.afinidad == "gravedad":
                def atraccion(s: Slot) -> float:
                    d = 1.0 - solapa(s) / x.prompt + 0.05
                    return r_s * (1 + s.cache) / (d * d) if solapa(s) else r_s / (d * d)
                slot = max(libres, key=atraccion)
            else:
                slot = max(libres, key=lambda s: (solapa(s), -s.id))
            reutil = solapa(slot)
            # al usar el slot se descarta la caché que no sirve
            m_sale += slot.cache - reutil
            slot.cache, slot.familia_cache = 0, -1
            if x.hibernada:         # restaurar su KV desde disco: conserva todo su progreso
                swap_ms += cfg.t_swap_por_token * x.kv
                x.hibernada = False
            else:
                x.prefill_hecho = reutil
            x.slot = slot.id
            if x.inicio < 0:
                x.inicio = t
            slot.tarea = x
            libres.remove(slot)
            cola.pop(i)

        # --- lote del paso ---
        activos = [s for s in slots if s.tarea]
        prefill_slots = [s for s in activos if s.tarea.prefill_hecho < s.tarea.prompt]
        if pol.prefill_corto_primero:
            prefill_slots.sort(key=lambda s: s.tarea.prompt - s.tarea.prefill_hecho)
        presupuesto = cfg.n_batch
        n_pre = 0
        for s in prefill_slots:
            if presupuesto <= 0:
                break
            n = min(presupuesto, s.tarea.prompt - s.tarea.prefill_hecho)
            s.tarea.prefill_hecho += n
            presupuesto -= n
            n_pre += n
        n_gen = 0
        for s in activos:
            x = s.tarea
            if x.prefill_hecho >= x.prompt:     # el último trozo de prefill ya produce el primer token
                if x.generados < x.salida:
                    x.generados += 1
                    n_gen += 1
                    if x.primer_token < 0:
                        x.primer_token = t
        # memoria en KV estático: context shift dentro del slot (no desborda, pero trunca)
        if pol.kv_estatico:
            for s in activos:
                x = s.tarea
                if x.kv > cap_slot:
                    quitar = x.kv - cap_slot
                    x.truncada = True           # el modelo pierde parte de su contexto
                    x.descartados += quitar
                    m_sale += quitar
        else:
            # memoria compartida: si se desborda, se expulsa una tarea (pierde su progreso)
            while kv_usado() > C:
                ociosos = [s for s in slots if not s.tarea and s.cache]
                if ociosos:
                    s = ociosos[0]
                    m_sale += s.cache
                    s.cache, s.familia_cache = 0, -1
                    continue
                activos = [s for s in slots if s.tarea]
                if pol.ordenar == "fifo":
                    victima = max(activos, key=lambda s: s.tarea.inicio)       # la más reciente
                else:
                    victima = max(activos, key=lambda s: prioridad(s.tarea))   # la de mayor costo
                expulsar(victima)

        # --- terminaciones ---
        for s in slots:
            x = s.tarea
            if x and x.prefill_hecho >= x.prompt and x.generados >= x.salida:
                x.fin = t
                terminadas += 1
                if pol.adaptar_masa:    # campo: la masa observada corrige la estimación
                    q = math.log(max(1, x.salida) / max(1, x.salida_est))
                    log_q += pol.kappa * (q - log_q)
                    margen = min(4.0, max(0.5, pol.margen_estimacion * math.exp(log_q)))
                res.tokens_generados += x.generados
                conserva = min(x.prefijo, x.kv) if not x.truncada else 0
                m_sale += x.kv - conserva
                s.tarea = None
                s.cache, s.familia_cache = conserva, x.familia

        for x in cola:
            x.espera_pasos += 1

        # --- Friedmann: tasa de expansión y anticipación del colapso ---
        m_entra = n_pre + n_gen
        flujo = (1 - pol.alfa_flujo) * flujo + pol.alfa_flujo * (m_entra - m_sale) / C
        rho = kv_activo() / C
        activos = [s for s in slots if s.tarea]
        exceso = sum(1 for s in activos if s.tarea.generados > margen * s.tarea.salida_est
                     and s.tarea.generados < s.tarea.salida)
        flujo_exc = (1 - pol.alfa_flujo) * flujo_exc + pol.alfa_flujo * exceso / C
        if pol.anticipar and not pol.kv_estatico and len(activos) > 1:
            # las tareas que ya superaron su estimación seguirán creciendo: si la proyección
            # rebasa ρ_max se hiberna la de mayor costo restante (nunca la única tarea activa)
            if (comprometido() + flujo_exc * C * pol.H) / C > pol.rho_max:
                victima = max(activos, key=lambda s: prioridad(s.tarea))
                expulsar(victima)
                rho = kv_activo() / C

        dt = (cfg.t_base + cfg.t_por_gen * n_gen + cfg.t_por_pre * n_pre + cfg.t_por_kv * kv_usado()
              + swap_ms)
        swap_ms = 0.0
        t += dt
        if paso % 10 == 0:
            res.serie_t.append(t / 1000.0)
            res.serie_rho.append(rho)
            res.serie_margen.append(margen)
            res.serie_cola.append(len(cola))

    res.t_total = t
    return res


# ---------------------------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------------------------


def pct(v: list[float], p: float) -> float:
    if not v:
        return float("nan")
    v = sorted(v)
    i = min(len(v) - 1, max(0, int(round(p / 100.0 * (len(v) - 1)))))
    return v[i]


def resumir(r: Resultado) -> dict:
    hechas = [x for x in r.tareas if x.fin >= 0]
    out = {"politica": r.politica, "completadas": len(hechas), "total": len(r.tareas),
           "fallidas": sum(x.fallida for x in r.tareas),
           "throughput_tok_s": r.tokens_generados / (r.t_total / 1000.0) if r.t_total else 0.0,
           "preempciones": r.preempciones, "tokens_perdidos": r.tokens_perdidos,
           "hibernaciones": r.hibernaciones,
           "truncadas": sum(x.truncada for x in r.tareas),
           "rho_max": max(r.serie_rho) if r.serie_rho else 0.0,
           "rho_sd": statistics.pstdev(r.serie_rho[len(r.serie_rho) // 5:]) if len(r.serie_rho) > 10 else 0.0}
    for clase in ("corta", "compleja"):
        lat = [(x.fin - x.llegada) / 1000.0 for x in hechas if x.clase == clase]
        ttft = [(x.primer_token - x.llegada) / 1000.0 for x in hechas if x.clase == clase and x.primer_token >= 0]
        out[f"{clase}_n"] = len(lat)
        out[f"{clase}_p50"] = pct(lat, 50)
        out[f"{clase}_p95"] = pct(lat, 95)
        out[f"{clase}_ttft_p50"] = pct(ttft, 50)
    return out


# Configuración común del flujo NAI: matriz S + tiempo emergente + horizonte + N cuerpos.
NAI_BASE = dict(ordenar="costo", envejecimiento_tau=400.0, horizonte=True, afinidad="gravedad",
                prefill_corto_primero=True)
# Márgenes estáticos mínimos (más memoria útil) + carril rápido de 2 slots.
NAI_AGRESIVA = dict(NAI_BASE, reserva=0.03, margen_estimacion=1.0, slots_rapidos=2)


def politicas_base() -> list[Politica]:
    return [
        Politica("llama.cpp KV fijo, 16 slots", kv_estatico=True),
        Politica("llama.cpp KV fijo, 4 slots", kv_estatico=True, slots_usables=4),
        Politica("llama.cpp KV unificado", ),
        Politica("Solo 'lo rápido primero'", ordenar="costo"),
        Politica("NAI agresivo (sin control)", **NAI_AGRESIVA),
        # recomendada: horizonte + campo (masa adaptativa) + hibernación como red de seguridad
        Politica("NAI completo", **NAI_AGRESIVA, hibernar=True, adaptar_masa=True),
        Politica("NAI completo + anticipación", **NAI_AGRESIVA, hibernar=True, adaptar_masa=True, anticipar=True),
    ]


# ---------------------------------------------------------------------------------------------
# Experimentos
# ---------------------------------------------------------------------------------------------

ESCENARIOS = {
    "mixto":           dict(n=600, tasa_por_s=1.8, frac_compleja=0.3),
    "rafaga":          dict(n=600, tasa_por_s=1.6, frac_compleja=0.3, rafaga=True),
    "puras_complejas": dict(n=250, tasa_por_s=0.6, frac_compleja=1.0),
    "subestimacion":   dict(n=600, tasa_por_s=1.8, frac_compleja=0.3, sesgo_est=0.33),
    "extremo":         dict(n=600, tasa_por_s=1.6, frac_compleja=0.3, rafaga=True, sesgo_est=0.33,
                            cola_pesada=True, sigma_est=0.9),
}
COLUMNAS = ["escenario", "politica", "completadas", "fallidas", "total", "corta_p50", "corta_p95", "corta_ttft_p50",
            "compleja_p50", "compleja_p95", "compleja_ttft_p50", "throughput_tok_s", "preempciones",
            "tokens_perdidos", "hibernaciones", "truncadas", "rho_max", "rho_sd"]


def promediar(lista: list[dict]) -> dict:
    out = {}
    for k, v in lista[0].items():
        if isinstance(v, (int, float)):
            vals = [x[k] for x in lista if not (isinstance(x[k], float) and math.isnan(x[k]))]
            out[k] = statistics.mean(vals) if vals else float("nan")
        else:
            out[k] = v
    return out


def experimentos(salida: Path, semillas: list[int]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    salida.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    filas = []
    series = {}
    for esc, params in ESCENARIOS.items():
        for pol in politicas_base():
            res_sem = []
            for s in semillas:
                r = simular(cfg, pol, generar_carga(semilla=s, **params))
                res_sem.append(resumir(r))
                if s == semillas[0]:
                    series[(esc, pol.nombre)] = r
            prom = promediar(res_sem)
            prom["escenario"] = esc
            filas.append(prom)
            print(f"  {esc:15s} {pol.nombre:28s} cortas p50 {prom['corta_p50']:6.1f}s p95 {prom['corta_p95']:6.1f}s | "
                  f"complejas p50 {prom['compleja_p50']:6.1f}s p95 {prom['compleja_p95']:6.1f}s | "
                  f"{prom['throughput_tok_s']:5.1f} tok/s | fallidas {prom['fallidas']:5.1f} "
                  f"| hib {prom['hibernaciones']:5.1f} | trunc {prom['truncadas']:5.1f}",
                  flush=True)

    with open(salida / "resultados.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for fila in filas:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in fila.items()})

    # --- barrido de estabilidad: anticipación H × inercia α (escenario subestimación) ---
    Hs = [0, 10, 20, 40, 80, 160]
    alfas = [0.02, 0.05, 0.1, 0.2, 0.5]
    carga = generar_carga(semilla=semillas[0], **ESCENARIOS["subestimacion"])
    mapas = {m: [[0.0] * len(alfas) for _ in Hs] for m in ("hibernaciones", "corta_p95", "compleja_p95", "throughput_tok_s")}
    with open(salida / "barrido_H_alfa.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["H", "alfa", "hibernaciones", "corta_p95_s", "compleja_p95_s", "throughput_tok_s"])
        for i, H in enumerate(Hs):
            for j, al in enumerate(alfas):
                pol = Politica("barrido", **NAI_AGRESIVA, hibernar=True, anticipar=H > 0, H=max(H, 1),
                               alfa_flujo=al, adaptar_masa=True)
                r = resumir(simular(cfg, pol, carga))
                for m in mapas:
                    mapas[m][i][j] = r[m]
                w.writerow([H, al, r["hibernaciones"], f"{r['corta_p95']:.2f}", f"{r['compleja_p95']:.2f}",
                            f"{r['throughput_tok_s']:.1f}"])

    # --- gráficas ---
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for nombre, color in [("llama.cpp KV unificado", "#D85A30"), ("NAI agresivo (sin control)", "#7F77DD"),
                          ("NAI completo", "#1D9E75")]:
        r = series[("subestimacion", nombre)]
        ax[0].plot(r.serie_t, r.serie_rho, label=nombre, color=color, lw=0.9)
        ax[1].plot(r.serie_t, r.serie_cola, label=nombre, color=color, lw=0.9)
        if nombre == "NAI completo":
            ax[2].plot(r.serie_t, r.serie_margen, color=color, lw=1.2, label="margen adaptativo (campo)")
    ax[0].axhline(1.0, color="#888780", lw=0.8, ls="--")
    ax[0].set_ylabel("ocupación ρ")
    ax[0].legend(loc="lower right", fontsize=8)
    ax[1].set_ylabel("tareas en cola")
    ax[2].axhline(3.0, color="#888780", lw=0.8, ls=":", label="margen ideal (1/0.33)")
    ax[2].set_ylabel("margen de masa")
    ax[2].legend(loc="lower right", fontsize=8)
    ax[2].set_xlabel("tiempo simulado [s]")
    ax[0].set_title("Escenario subestimación: el estimador predice 1/3 de la salida real")
    fig.tight_layout()
    fig.savefig(salida / "serie_subestimacion.png", dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(1, 4, figsize=(18, 4.2))
    for a, (m, titulo) in zip(axs, [("hibernaciones", "hibernaciones"), ("corta_p95", "latencia p95 cortas [s]"),
                                    ("compleja_p95", "latencia p95 complejas [s]"),
                                    ("throughput_tok_s", "throughput [tok/s]")]):
        im = a.imshow(mapas[m], aspect="auto", origin="lower", cmap="viridis")
        a.set_xticks(range(len(alfas)), [str(x) for x in alfas])
        a.set_yticks(range(len(Hs)), [str(x) for x in Hs])
        a.set_xlabel("inercia α (suavizado del flujo)")
        a.set_ylabel("anticipación H [pasos]")
        a.set_title(titulo, fontsize=10)
        for i in range(len(Hs)):
            for j in range(len(alfas)):
                a.text(j, i, f"{mapas[m][i][j]:.0f}", ha="center", va="center", fontsize=7, color="white")
        fig.colorbar(im, ax=a)
    fig.suptitle("Barrido del amortiguamiento por flujo de masa (H = 0: sin anticipación)", fontsize=11)
    fig.tight_layout()
    fig.savefig(salida / "barrido_H_alfa.png", dpi=130)
    plt.close(fig)

    graficar_resultados(salida)
    print(f"\n[NAI] resultados en {salida}/")


def graficar_resultados(salida: Path) -> None:
    """Resumen por escenario desde resultados.csv: integridad, latencia p95 y throughput."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(salida / "resultados.csv") as f:
        filas = list(csv.DictReader(f))
    escenarios = list(dict.fromkeys(f["escenario"] for f in filas))
    nombres = list(dict.fromkeys(f["politica"] for f in filas))
    colores = ["#B4B2A9", "#888780", "#D85A30", "#F0997B", "#AFA9EC", "#1D9E75", "#5DCAA5"]
    fig, axs = plt.subplots(3, len(escenarios), figsize=(4.2 * len(escenarios), 10), squeeze=False)
    for j, esc in enumerate(escenarios):
        fs = {f["politica"]: f for f in filas if f["escenario"] == esc}
        x = list(range(len(nombres)))
        total = [float(fs[n]["total"]) for n in nombres]
        integras = [100.0 * (float(fs[n]["completadas"]) - float(fs[n]["truncadas"])) / t for n, t in zip(nombres, total)]
        lat = [float(fs[n]["compleja_p95"]) for n in nombres]
        cor = [float(fs[n]["corta_p95"]) for n in nombres]
        tps = [float(fs[n]["throughput_tok_s"]) for n in nombres]
        axs[0][j].bar(x, integras, color=colores[:len(nombres)])
        axs[0][j].set_ylim(0, 105)
        axs[0][j].set_title(esc.replace("_", " "), fontsize=11)
        for i, v in enumerate(integras):
            axs[0][j].text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=7)
        axs[1][j].bar([i - 0.2 for i in x], cor, 0.4, color="#1D9E75", label="cortas")
        axs[1][j].bar([i + 0.2 for i in x], lat, 0.4, color="#7F77DD", label="complejas")
        axs[1][j].set_yscale("log")
        axs[2][j].bar(x, tps, color=colores[:len(nombres)])
        for fila in axs:
            fila[j].set_xticks(x, [""] * len(x))
        axs[2][j].set_xticks(x, nombres, rotation=40, ha="right", fontsize=7)
    axs[0][0].set_ylabel("% completas sin error\nni contexto truncado")
    axs[1][0].set_ylabel("latencia p95 [s]\n(solo completadas)")
    axs[1][0].legend(fontsize=8)
    axs[2][0].set_ylabel("throughput [tok/s]")
    fig.suptitle("Simulación: llama.cpp vs. flujo NAI (16 slots, 32k tokens de KV)", fontsize=12)
    fig.tight_layout()
    fig.savefig(salida / "resumen.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Trazas reales (llama-server --nai-traza)
# ---------------------------------------------------------------------------------------------


def cargar_traza(ruta: Path, sigma_est: float) -> tuple[list[Tarea], dict]:
    """Lee PREFIJO-tareas.csv de llama-server --nai-traza. Devuelve la carga y la latencia real."""
    rnd = random.Random(0)
    tareas, lat_real = [], {"corta": [], "compleja": []}
    with open(ruta) as f:
        filas = list(csv.DictReader(f))
    t0 = min(float(x["t_llegada_ms"]) for x in filas)
    for i, fila in enumerate(filas):
        salida, prompt = int(fila["salida"]), int(fila["prompt"])
        clase = "compleja" if prompt + salida > 2000 else "corta"
        tareas.append(Tarea(i, float(fila["t_llegada_ms"]) - t0, 0, int(fila.get("cacheados", 0)), prompt, salida,
                            max(1, int(salida * math.exp(rnd.gauss(0, sigma_est)))), clase))
        lat_real[clase].append((float(fila["t_fin_ms"]) - float(fila["t_llegada_ms"])) / 1000.0)
    real = {f"{c}_p50": pct(v, 50) for c, v in lat_real.items()}
    real.update({f"{c}_p95": pct(v, 95) for c, v in lat_real.items()})
    return sorted(tareas, key=lambda x: x.llegada), real


def calibrar(ruta: Path) -> Config:
    """Ajusta el modelo de tiempo por paso con mínimos cuadrados sobre pasos.csv."""
    import numpy as np
    X, y = [], []
    with open(ruta) as f:
        for fila in csv.DictReader(f):
            X.append([1.0, float(fila["tokens_gen"]), float(fila["tokens_prefill"]), float(fila["kv_usado"])])
            y.append(float(fila["dt_ms"]))
    coef, *_ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)
    cfg = Config(t_base=max(0.0, coef[0]), t_por_gen=max(0.0, coef[1]), t_por_pre=max(0.0, coef[2]),
                 t_por_kv=max(0.0, coef[3]))
    print(f"[NAI] modelo calibrado: dt = {cfg.t_base:.2f} + {cfg.t_por_gen:.3f}·gen + "
          f"{cfg.t_por_pre:.4f}·prefill + {cfg.t_por_kv:.6f}·kv  [ms]")
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description="[NAI] simulador del planificador entrelazado")
    ap.add_argument("--salida", type=Path, default=Path("nai-sim"))
    ap.add_argument("--semillas", type=int, default=3)
    ap.add_argument("--traza", type=Path, help="tareas.csv de llama-server --nai-traza")
    ap.add_argument("--calibrar", type=Path, help="pasos.csv de llama-server --nai-traza")
    ap.add_argument("--capacidad", type=int, default=Config.capacidad)
    ap.add_argument("--slots", type=int, default=Config.slots)
    ap.add_argument("--graficar", type=Path, help="solo regenerar gráficas desde DIR/resultados.csv")
    a = ap.parse_args()

    if a.graficar:
        graficar_resultados(a.graficar)
        return 0

    if a.traza:
        cfg = calibrar(a.calibrar) if a.calibrar else Config()
        cfg.capacidad, cfg.slots = a.capacidad, a.slots
        carga, real = cargar_traza(a.traza, 0.5)
        print(f"  {'REAL (medido en la traza)':30s} cortas p95 {real['corta_p95']:7.1f}s | complejas p95 "
              f"{real['compleja_p95']:7.1f}s   <- compárese con la política de llama.cpp que usó el servidor")
        for pol in politicas_base():
            r = resumir(simular(cfg, pol, carga))
            print(f"  {pol.nombre:30s} cortas p95 {r['corta_p95']:7.1f}s | complejas p95 {r['compleja_p95']:7.1f}s | "
                  f"fallidas {r['fallidas']} | trunc {r['truncadas']}")
        return 0

    experimentos(a.salida, list(range(1, a.semillas + 1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
