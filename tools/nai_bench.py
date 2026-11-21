#!/usr/bin/env python3
# [NAI] Banco de pruebas: llama.cpp vanilla vs. cada función NAI, en el MISMO equipo y modelo.
#
# Mide, para cada variante, levantando llama-server con sus flags:
#   - velocidad de generación (tok/s) y de prefill, promedio de varias repeticiones;
#   - si la salida greedy (temp 0) es IDÉNTICA a la de vanilla (debe serlo con la autopista
#     y con la auto-especulación; con desalojo/gravedad puede cambiar y se informa);
#   - prueba de "aguja" en contexto largo para desalojo y gravedad: un dato al inicio del
#     prompt que el modelo debe repetir después de generar más de lo que cabe en el contexto.
#
# Uso:
#   python3 tools/nai/nai_bench.py -m modelo.gguf [--server build/bin/llama-server]
#          [--borrador borrador.gguf] [--reps 3] [--n-predict 128] [--ngl 99] [--salida res.json]
#
# Genera una tabla Markdown lista para pegar en el README y, opcionalmente, un JSON.
#
# Ruido: cada ronda levanta todas las variantes en orden ROTADO (ninguna corre siempre primero,
# con la máquina más fresca) y se reportan MEDIANAS y el costo en ms/token, comparable entre
# máquinas. Con --server-oficial se agrega el llama-server del llama.cpp oficial como variante,
# para medir el costo de NAI con todas sus funciones apagadas.
#
# Modo carga mixta (calibración del planificador):
#   python3 tools/nai/nai_bench.py -m modelo.gguf --carga-mixta 120 --slots 8 --ctx 32768
# Lanza peticiones concurrentes con la misma distribución que el simulador (cortas y complejas,
# prefijos compartidos) contra un llama-server con --nai-traza, y deja las trazas listas para:
#   python3 tools/nai/sim/nai_planificador_sim.py --traza traza-tareas.csv --calibrar traza-pasos.csv

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]

PROMPTS = [
    "Explica en un párrafo qué es una cuenta de ahorro y para qué sirve.",
    "Escribe una función en Python que invierta una cadena y explica su complejidad.",
    "Resume las causas principales de la Revolución Mexicana en cinco puntos.",
    "List three differences between TCP and UDP.",
]

AGUJA = "ZAFIRO-7342"


def puerto_libre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(url: str, cuerpo: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(cuerpo).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Servidor:
    def __init__(self, binario: str, modelo: str, extra: list[str], ctx: int, ngl: int, log: Path,
                 hilos: int = 0):
        self.puerto = puerto_libre()
        self.url = f"http://127.0.0.1:{self.puerto}"
        cmd = [binario, "-m", modelo, "--port", str(self.puerto), "-c", str(ctx), "-np", "1",
               "-ngl", str(ngl), "--no-webui", *(["-t", str(hilos)] if hilos > 0 else []), *extra]
        self.log = open(log, "w")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        self.cmd = cmd

    def esperar(self, limite: float = 300) -> None:
        t0 = time.time()
        while time.time() - t0 < limite:
            if self.proc.poll() is not None:
                raise RuntimeError(f"el servidor terminó (código {self.proc.returncode}); ver {self.log.name}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(0.5)
        raise TimeoutError("el servidor no respondió a tiempo")

    def completar(self, prompt: str, n_predict: int, extra: dict | None = None) -> dict:
        cuerpo = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "seed": 42,
                  "cache_prompt": False, "ignore_eos": True}
        cuerpo.update(extra or {})
        return post(self.url + "/completion", cuerpo)

    def cerrar(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


def prompt_aguja(n_relleno: int) -> str:
    relleno = " ".join(f"Nota {i}: el inventario de la bodega se revisa cada semana." for i in range(n_relleno))
    return (f"Dato importante: el código secreto es {AGUJA}. Recuérdalo.\n\n{relleno}\n\n"
            "Ahora escribe un texto largo sobre logística y, al final, repite el código secreto.")


def medir(srv: Servidor, reps: int, n_predict: int) -> dict:
    # Una petición con error (p. ej. HTTP 500 del parser de respuestas con texto que no es UTF-8
    # válido) se cuenta como error y no entra en las medianas; la corrida continúa.
    gen, pre, textos, errores = [], [], [], 0
    try:
        srv.completar("Hola", 8)  # calentamiento
    except urllib.error.HTTPError:
        pass
    for p in PROMPTS:
        for r in range(reps):
            try:
                res = srv.completar(p, n_predict)
            except urllib.error.HTTPError:
                errores += 1
                if r == 0:
                    textos.append(None)
                continue
            t = res.get("timings", {})
            gen.append(t.get("predicted_per_second", 0.0))
            pre.append(t.get("prompt_per_second", 0.0))
            if r == 0:
                textos.append(res.get("content", ""))
    return {"gen": gen, "pre": pre, "textos": textos, "errores": errores}


def cuartiles(v: list[float]) -> tuple[float, float, float]:
    v = sorted(v)
    q = statistics.quantiles(v, n=4) if len(v) >= 4 else [v[0], statistics.median(v), v[-1]]
    return q[0], statistics.median(v), q[2]


def carga_mixta(a) -> int:
    sys.path.insert(0, str(RAIZ / "tools" / "nai" / "sim"))
    from nai_planificador_sim import generar_carga  # misma distribución que el simulador

    tareas = generar_carga(n=a.carga_mixta, tasa_por_s=a.tasa, frac_compleja=a.frac_compleja, semilla=7)
    # ~1 token por palabra corta; la traza registra los tamaños reales tokenizados
    def texto(n: int, semilla: int) -> str:
        palabras = ["dato", "caso", "red", "nodo", "masa", "flujo", "orden", "tarea", "carga", "memoria"]
        return " ".join(palabras[(semilla * 31 + i * 7) % len(palabras)] for i in range(max(1, n)))

    prefijo = str(a.logs / "traza")
    flags = ["-np", str(a.slots), "--nai-traza", prefijo, *a.flags_servidor.split()]
    srv = Servidor(a.server, a.modelo, flags, a.ctx, a.ngl, a.logs / "carga-mixta.log")
    errores = 0
    try:
        srv.esperar()
        t0 = time.time()

        def lanzar(x) -> bool:
            espera = x.llegada / 1000.0 / a.escala_tiempo - (time.time() - t0)
            if espera > 0:
                time.sleep(espera)
            prompt = texto(x.prefijo, x.familia) + "\n" + texto(x.prompt - x.prefijo, x.id)
            try:
                srv.completar(prompt, x.salida, {"cache_prompt": True})
                return True
            except urllib.error.HTTPError:
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(256, len(tareas))) as ex:
            for ok in ex.map(lanzar, tareas):
                errores += 0 if ok else 1
    finally:
        srv.cerrar()
    print(f"[NAI] carga mixta: {len(tareas)} peticiones, {errores} con error")
    print(f"  trazas: {prefijo}-pasos.csv y {prefijo}-tareas.csv")
    print(f"  calibrar y simular: python3 tools/nai/sim/nai_planificador_sim.py --traza {prefijo}-tareas.csv "
          f"--calibrar {prefijo}-pasos.csv --capacidad {a.ctx} --slots {a.slots}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="[NAI] vanilla vs. NAI")
    ap.add_argument("-m", "--modelo", required=True)
    ap.add_argument("--server", default=str(RAIZ / "build" / "bin" / "llama-server"))
    ap.add_argument("--server-oficial", help="llama-server del llama.cpp oficial (agrega la variante 'llama.cpp oficial')")
    ap.add_argument("--rondas", type=int, default=3, help="rondas con orden rotado de variantes")
    ap.add_argument("--hilos", type=int, default=0, help="-t para llama-server (0 = automático)")
    ap.add_argument("--variantes", default="", help="lista separada por comas para correr solo algunas")
    ap.add_argument("--borrador", help="GGUF creado con nai_borrador_capas.py (activa la variante auto-especulación)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--ctx-aguja", type=int, default=1024, help="contexto pequeño para forzar context shift")
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--gravedad", type=float, default=5e-4)
    ap.add_argument("--salida", type=Path)
    ap.add_argument("--logs", type=Path, default=Path("nai-bench-logs"))
    ap.add_argument("--carga-mixta", type=int, default=0, help="N peticiones concurrentes (modo calibración)")
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--tasa", type=float, default=0.5, help="peticiones por segundo (carga mixta)")
    ap.add_argument("--frac-compleja", type=float, default=0.3)
    ap.add_argument("--escala-tiempo", type=float, default=1.0, help=">1 comprime las llegadas")
    ap.add_argument("--flags-servidor", default="", help="flags extra para llama-server; usa la forma --flags-servidor=\"--kv-unified ...\"")
    a = ap.parse_args()

    a.logs.mkdir(parents=True, exist_ok=True)
    if a.carga_mixta > 0:
        return carga_mixta(a)
    # (nombre, flags, binario)
    variantes = [
        ("vanilla", [], a.server),
        ("autopista", ["--nai-autopista"], a.server),
        ("desalojo", ["--context-shift", "--nai-desalojo"], a.server),
        ("gravedad", ["--nai-gravedad", str(a.gravedad)], a.server),
    ]
    if a.server_oficial:
        variantes.insert(0, ("llama.cpp oficial", [], a.server_oficial))
    if a.borrador:
        variantes.append(("auto-especulación", ["-md", a.borrador, "--spec-draft-n-max", "8", "--spec-draft-n-min", "1",
                                                "-ngld", str(a.ngl)], a.server))
    if a.variantes:
        pedidas = {v.strip() for v in a.variantes.split(",")}
        variantes = [v for v in variantes if v[0] in pedidas or v[0] == "vanilla"]

    resultados = {n: {"gen": [], "pre": [], "textos": None, "errores": 0} for n, _, _ in variantes}
    for ronda in range(a.rondas):
        k = ronda % len(variantes)
        orden = variantes[k:] + variantes[:k]   # orden rotado: nadie corre siempre primero
        for nombre, flags, binario in orden:
            print(f"[NAI] ronda {ronda + 1}/{a.rondas} · {nombre}: {' '.join(flags) or '(sin flags)'}", flush=True)
            srv = Servidor(binario, a.modelo, flags, a.ctx, a.ngl, a.logs / f"{nombre}-r{ronda + 1}.log", a.hilos)
            try:
                srv.esperar()
                m = medir(srv, a.reps, a.n_predict)
            finally:
                srv.cerrar()
            r = resultados[nombre]
            r["gen"] += m["gen"]
            r["pre"] += m["pre"]
            r["errores"] += m["errores"]
            if r["textos"] is None:
                r["textos"] = m["textos"]

    # aguja (una vez): contexto pequeño + generación larga => fuerza context shift
    for nombre, flags, binario in variantes:
        if nombre not in ("vanilla", "desalojo", "gravedad"):
            continue
        extra = ["--context-shift"] if nombre != "desalojo" else []
        srv = Servidor(binario, a.modelo, flags + extra, a.ctx_aguja, a.ngl, a.logs / f"{nombre}-aguja.log", a.hilos)
        try:
            srv.esperar()
            res = srv.completar(prompt_aguja(a.ctx_aguja // 40), a.ctx_aguja, {"n_keep": 0})
            resultados[nombre]["aguja"] = AGUJA in res.get("content", "")
        except urllib.error.HTTPError as e:
            resultados[nombre]["aguja"] = f"error HTTP {e.code}"
        finally:
            srv.cerrar()

    resultados = {n: r for n, r in resultados.items() if r["gen"]}
    if "vanilla" not in resultados:
        print("[NAI] todas las peticiones de vanilla fallaron; revisa los logs")
        return 1
    base = resultados["vanilla"]
    _, base_med, _ = cuartiles(base["gen"])
    base_ms = 1000.0 / base_med
    print("\n| Variante | Generación mediana (tok/s) | Rango intercuartil | ms/token | Δ ms/token | Δ % | "
          "Prefill mediana (tok/s) | Salida greedy = NAI vanilla | Errores | Aguja |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for nombre, r in resultados.items():
        q1, med, q3 = cuartiles(r["gen"])
        ms = 1000.0 / med
        pares = [(x, y) for x, y in zip(r["textos"], base["textos"]) if x is not None and y is not None]
        iguales = sum(x == y for x, y in pares)
        aguja = r.get("aguja", "—")
        aguja = "sí" if aguja is True else "no" if aguja is False else aguja
        r.update({"gen_mediana": med, "ms_token": ms, "delta_ms": ms - base_ms, "iguales": iguales})
        print(f"| {nombre} | {med:.2f} | {q1:.1f}–{q3:.1f} | {ms:.2f} | {ms - base_ms:+.2f} | "
              f"{100.0 * (med / base_med - 1.0):+.1f}% | {statistics.median(r['pre']):.1f} | "
              f"{iguales}/{len(pares)} | {r['errores']} | {aguja} |")
    n_med = a.rondas * a.reps * len(PROMPTS)
    print(f"\nModelo: {os.path.basename(a.modelo)} | {a.rondas} rondas × {a.reps} reps × {len(PROMPTS)} prompts "
          f"= {n_med} mediciones por variante | n_predict: {a.n_predict} | ngl: {a.ngl}")
    print("Δ ms/token = costo absoluto por token frente a NAI vanilla (todas las funciones apagadas).")
    print("Si el rango intercuartil de dos variantes se traslapa, la diferencia no es concluyente.")

    if a.salida:
        a.salida.write_text(json.dumps(resultados, ensure_ascii=False, indent=2))
        print(f"JSON: {a.salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
