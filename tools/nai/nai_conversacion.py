#!/usr/bin/env python3
"""[NAI] Conversación larga contra un llama-server: el cliente reenvía TODO el historial en cada turno
(como la interfaz web). Por turno mide: si responde o falla, tokens de prompt evaluados, tiempo
de prefill, velocidad de generación y latencia total. Levanta y detiene el servidor él mismo.

Uso (oficial vs NAI, mismo modelo):
  python3 tools/nai/nai_conversacion.py --server ../llama.cpp/build/bin/llama-server -m m.gguf --nombre oficial --salida oficial.json
  python3 tools/nai/nai_conversacion.py --server build/bin/llama-server -m m.gguf --nombre nai-memoria \
          --flags "--nai-memoria --nai-sesiones DIR" --salida nai.json
(DIR se reemplaza por una carpeta temporal propia de la corrida.)"""
import json, subprocess, sys, time, urllib.request, urllib.error, argparse, os, signal

ap = argparse.ArgumentParser()
ap.add_argument("--server", required=True)
ap.add_argument("-m", "--modelo", required=True)
ap.add_argument("--nombre", required=True)
ap.add_argument("--turnos", type=int, default=20)
ap.add_argument("--ctx", type=int, default=2048)
ap.add_argument("--puerto", type=int, default=8111)
ap.add_argument("--flags", default="")
ap.add_argument("--salida", required=True)
a = ap.parse_args()

TEMAS = ["la milpa y el maíz criollo", "el ciclo del agua en la sierra", "la historia de Monterrey",
         "cómo ahorrar cada quincena", "la filosofía estoica", "el cuidado de un perro viejo",
         "las cosmogonías mesoamericanas", "la física de un puente colgante", "la medicina de las abuelas",
         "el silencio en las tradiciones contemplativas"]

url = f"http://localhost:{a.puerto}"
d = f"/tmp/conv-{a.nombre}"
subprocess.run(["rm", "-rf", d]); os.makedirs(d)
cmd = [a.server, "-m", a.modelo, "-c", str(a.ctx), "-np", "1", "-t", "2", "--port", str(a.puerto), "--no-webui"]
cmd += [x.replace("DIR", d) for x in a.flags.split()]
log = open(f"{d}/server.log", "w")
srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
try:
    for _ in range(180):
        try:
            if json.load(urllib.request.urlopen(url + "/health", timeout=2)).get("status") == "ok": break
        except Exception: time.sleep(1)
    msgs, filas = [], []
    for i in range(a.turnos):
        tema = TEMAS[i % len(TEMAS)]
        msgs.append({"role": "user", "content": f"Turno {i + 1}. Cuéntame con detalle sobre {tema}, "
                     "con ejemplos concretos, y relaciónalo con lo que hablamos antes en esta conversación."})
        cuerpo = {"messages": msgs, "max_tokens": 64, "temperature": 0, "ignore_eos": True}
        t0 = time.time()
        req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(cuerpo).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=900))
            dt = time.time() - t0
            tm = r.get("timings", {})
            texto = r["choices"][0]["message"]["content"]
            msgs.append({"role": "assistant", "content": texto})
            filas.append({"turno": i + 1, "ok": True, "prompt_n": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms"),
                          "tok_s": tm.get("predicted_per_second"), "latencia_s": round(dt, 2),
                          "truncated": r.get("truncated", False)})
        except urllib.error.HTTPError as e:
            filas.append({"turno": i + 1, "ok": False, "error": e.read().decode()[:160], "latencia_s": round(time.time() - t0, 2)})
            msgs.pop()  # el cliente no recibe respuesta; el usuario vuelve a intentar con otro tema
        print(a.nombre, json.dumps(filas[-1], ensure_ascii=False), flush=True)
        time.sleep(1)
    json.dump({"nombre": a.nombre, "flags": a.flags, "ctx": a.ctx, "filas": filas}, open(a.salida, "w"), ensure_ascii=False, indent=1)
finally:
    srv.send_signal(signal.SIGINT)
    try: srv.wait(20)
    except Exception: srv.kill()
