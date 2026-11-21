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
    def __init__(self, binario: str, modelo: str, extra: list[str], ctx: int, ngl: int, log: Path):
        self.puerto = puerto_libre()
        self.url = f"http://127.0.0.1:{self.puerto}"
        cmd = [binario, "-m", modelo, "--port", str(self.puerto), "-c", str(ctx), "-np", "1",
               "-ngl", str(ngl), "--no-webui", *extra]
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
    gen, pre, textos = [], [], []
    srv.completar("Hola", 8)  # calentamiento
    for p in PROMPTS:
        for r in range(reps):
            res = srv.completar(p, n_predict)
            t = res.get("timings", {})
            gen.append(t.get("predicted_per_second", 0.0))
            pre.append(t.get("prompt_per_second", 0.0))
            if r == 0:
                textos.append(res.get("content", ""))
    return {"gen_tps": statistics.mean(gen), "gen_tps_sd": statistics.pstdev(gen),
            "pre_tps": statistics.mean(pre), "textos": textos}


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
    variantes = [
        ("vanilla", []),
        ("autopista", ["--nai-autopista"]),
        ("desalojo", ["--context-shift", "--nai-desalojo"]),
        ("gravedad", ["--nai-gravedad", str(a.gravedad)]),
    ]
    if a.borrador:
        variantes.append(("auto-especulación", ["-md", a.borrador, "--spec-draft-n-max", "8", "--spec-draft-n-min", "1",
                                                "-ngld", str(a.ngl)]))

    resultados = {}
    for nombre, flags in variantes:
        print(f"[NAI] variante {nombre}: {' '.join(flags) or '(sin flags)'}", flush=True)
        srv = Servidor(a.server, a.modelo, flags, a.ctx, a.ngl, a.logs / f"{nombre}.log")
        try:
            srv.esperar()
            resultados[nombre] = medir(srv, a.reps, a.n_predict)
        finally:
            srv.cerrar()

        # aguja: contexto pequeño + generación larga => fuerza context shift
        if nombre in ("vanilla", "desalojo", "gravedad"):
            extra = ["--context-shift"] if nombre != "desalojo" else []
            srv = Servidor(a.server, a.modelo, flags + extra, a.ctx_aguja, a.ngl, a.logs / f"{nombre}-aguja.log")
            try:
                srv.esperar()
                res = srv.completar(prompt_aguja(a.ctx_aguja // 40), a.ctx_aguja, {"n_keep": 0})
                resultados[nombre]["aguja"] = AGUJA in res.get("content", "")
            except urllib.error.HTTPError as e:
                resultados[nombre]["aguja"] = f"error HTTP {e.code}"
            finally:
                srv.cerrar()

    base = resultados["vanilla"]
    print("\n| Variante | Generación (tok/s) | Δ vs vanilla | Prefill (tok/s) | Salida greedy = vanilla | Aguja |")
    print("|---|---|---|---|---|---|")
    for nombre, r in resultados.items():
        delta = 100.0 * (r["gen_tps"] / base["gen_tps"] - 1.0) if base["gen_tps"] else 0.0
        iguales = sum(x == y for x, y in zip(r["textos"], base["textos"]))
        aguja = r.get("aguja", "—")
        aguja = "sí" if aguja is True else "no" if aguja is False else aguja
        print(f"| {nombre} | {r['gen_tps']:.2f} ± {r['gen_tps_sd']:.2f} | {delta:+.1f}% | {r['pre_tps']:.1f} "
              f"| {iguales}/{len(base['textos'])} | {aguja} |")
    print(f"\nModelo: {os.path.basename(a.modelo)} | reps: {a.reps} | n_predict: {a.n_predict} | ngl: {a.ngl}")
    print("Nota: con un modelo de prueba aleatorio la 'aguja' no es significativa; usa un modelo real.")

    if a.salida:
        a.salida.write_text(json.dumps(resultados, ensure_ascii=False, indent=2))
        print(f"JSON: {a.salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
