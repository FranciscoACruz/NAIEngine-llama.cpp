#!/usr/bin/env python3
"""[NAI] Reentrenamiento por currículo: consolidar la biblioteca en un adaptador LoRA.

La memoria NAI (--nai-memoria) recuerda sin tocar el modelo. Este proceso da el paso
siguiente: lo que se aprendió en la BIBLIOTECA (documentos, nunca conversaciones de usuarios)
se consolida en los pesos mediante un adaptador LoRA, como el cerebro pasa recuerdos del
hipocampo a la corteza.

Currículo (la identidad y los temas prioritarios primero):
  fase 1  fundamentos: la carta de identidad (docs/IDENTIDAD_NAI.md, la prioridad más alta: se
          recorre dos veces por época), la carta de prioridades (qué es cada tema, por qué importa, cómo se
          relaciona) + todo lo aprendido de las 19 regiones de peso 3. Se recorre `peso` veces
          (3 épocas por defecto).
  fase 2  temas generales + REPASO de los prioritarios (35 % por defecto) para no olvidarlos.

Antes de aceptar el adaptador se mide la pérdida por tema (base vs. adaptador) y la deriva
sobre respuestas del modelo base (control de olvido). Si un tema empeora más de la
tolerancia, el adaptador se rechaza.

Pasos (cada uno se puede correr por separado):
  fuentes    busca en la red las consultas de docs/nai-fuentes.txt (SearxNG o Brave), filtra los
             resultados y los carga en la biblioteca del servidor (POST /nai/aprender)
  exportar   reconstruye la biblioteca a JSONL (nai_memoria --exportar)
  datos      arma los ejemplos: memoria literal, preguntas ancladas en la fuente, clasificación,
             carta de prioridades, conjuntos de prueba por tema y control de olvido
  entrenar   MLX (Apple Silicon): fase 1 y fase 2 con el mismo adaptador
  medir      pérdida por tema y control de olvido, base vs. adaptador; decide si se acepta
  convertir  MLX -> PEFT -> GGUF (convert_lora_to_gguf.py) para usar con llama-server --lora
  todo       exportar, datos, entrenar, medir y, si se acepta, convertir

Requisitos (Mac con Apple Silicon):
  pip install mlx-lm safetensors numpy
  pip install -r requirements/requirements-convert_lora_to_gguf.txt     # para convertir

Guardián: cada recuerdo exportado trae su naturaleza. La manipulación nunca se entrena; el
desahogo se entrena solo con su contexto («una persona bajo presión, no una verdad sobre la
humanidad») y con peso bajo; la sabiduría entra a la fase 1 con los prioritarios.

Importante: los recuerdos se guardan por defecto con el diccionario de tokens (no dependen de
las probabilidades del modelo). Con --nai-memoria-compacta sí dependen de las probabilidades
EXACTAS del modelo base: no reemplaces el .gguf base por un modelo fusionado.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]

# Mismas regiones que common/nai-memoria.cpp (peso, nombre)
REGIONES = [
    (3, "conciencia"), (3, "ética"), (3, "espiritualidad"), (3, "empatía"), (3, "psique"),
    (3, "filosofía"), (3, "lógica"), (3, "matemáticas"), (3, "física"), (3, "química"),
    (3, "biología"), (3, "ingeniería"), (3, "mecánica"), (3, "aerodinámica"), (3, "ciberseguridad"),
    (3, "supervivencia"), (3, "agricultura"), (3, "arte"), (3, "medicina tradicional"),
    (1, "finanzas personales"), (1, "cocina"), (1, "compras"), (1, "deportes"), (1, "trámites"),
    (1, "entretenimiento"), (1, "general"),
]

RELLENO = set("""para pero como porque cuando donde sobre entre hasta desde este esta estos estas esto
ella ellos ellas tiene tienen tener hace hacer puede pueden debe deben sido estar estaba fueron seria solo
tambien todo todos toda todas cada otro otra otros otras mismo misma algo nada mucho mucha muchos poco pocos
menos bien aqui entonces luego despues antes ahora siempre nunca segun mientras aunque sino pues cual cuales
quien quienes donde dijo decir dice texto tema puede puedo quiero sobre según""".split())


# ---------------------------------------------------------------------------- utilidades
def log(msg: str) -> None:
    print(msg, flush=True)


def normalizar(w: str) -> str:
    w = unicodedata.normalize("NFD", w.lower())
    return "".join(c for c in w if unicodedata.category(c) != "Mn")


def palabras_contenido(texto: str) -> list[str]:
    res = []
    for w in re.findall(r"\w+", texto, flags=re.UNICODE):
        n = normalizar(w)
        if len(n) >= 4 and n not in RELLENO and not n.isdigit():
            res.append(n[:6])
    return res


def leer_jsonl(ruta: Path) -> list[dict]:
    if not ruta.exists():
        return []
    return [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]


def escribir_jsonl(ruta: Path, filas: list[dict]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8") as f:
        for r in filas:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def leer_regiones(ruta: str | None) -> list[tuple[int, str]]:
    if not ruta:
        return REGIONES
    res = []
    for l in Path(ruta).read_text(encoding="utf-8").splitlines():
        if not l.strip() or l.startswith("#"):
            continue
        partes = l.split("|")
        if len(partes) >= 2:
            res.append((max(1, int(partes[0] or 1)), partes[1].strip()))
    return res


def http_json(url: str, cuerpo: dict | None = None, timeout: float = 600) -> dict:
    datos = None if cuerpo is None else json.dumps(cuerpo).encode("utf-8")
    req = urllib.request.Request(url, data=datos, headers={"Content-Type": "application/json",
                                                           "User-Agent": "NAI-reentrenar/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:  # mostrar el motivo que da el servidor
        try:
            msg = json.loads(e.read().decode("utf-8")).get("error", {}).get("message", "")
        except Exception:
            msg = ""
        raise RuntimeError(f"HTTP {e.code}: {msg or e.reason}") from None


def mensajes(usuario: str, asistente: str, sistema: str | None = None) -> dict:
    m = []
    if sistema:
        m.append({"role": "system", "content": sistema})
    m += [{"role": "user", "content": usuario}, {"role": "assistant", "content": asistente}]
    return {"messages": m}


# ---------------------------------------------------------------------------- fuentes (búsquedas en red)
# Sitios que no se aprenden: redes sociales y video (poco texto, contenido de terceros), tiendas,
# sitios de tareas o documentos subidos por usuarios (derechos de autor dudosos) y enciclopedias
# colaborativas (decisión del autor: el modelo aprende de búsquedas en red, no de Wikipedia).
BLOQUEADOS = {
    "facebook.com", "instagram.com", "tiktok.com", "youtube.com", "youtu.be", "x.com", "twitter.com",
    "threads.net", "pinterest.com", "linkedin.com", "reddit.com", "quora.com", "spotify.com",
    "amazon.com", "amazon.com.mx", "mercadolibre.com", "mercadolibre.com.mx", "ebay.com", "aliexpress.com",
    "scribd.com", "slideshare.net", "studocu.com", "coursehero.com", "brainly.lat", "brainly.com",
    "monografias.com", "rincondelvago.com", "wikipedia.org", "wikiwand.com", "wikimedia.org", "wiki2.org",
}


def dominio(url: str) -> str:
    d = urllib.parse.urlsplit(url).hostname or ""
    return d[4:] if d.startswith("www.") else d


def bloqueado(url: str, extra: set[str], permitidos: set[str]) -> bool:
    d = dominio(url)
    if permitidos:
        return not any(d == p or d.endswith("." + p) for p in permitidos)
    return any(d == b or d.endswith("." + b) for b in BLOQUEADOS | extra)


def leer_lista(ruta: str | None) -> set[str]:
    if not ruta:
        return set()
    return {l.strip().lower() for l in Path(ruta).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")}


def http_get_json(url: str, cabeceras: dict | None = None, timeout: float = 30) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "NAI-reentrenar/1.0", "Accept": "application/json",
                                               **(cabeceras or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def buscar(a, consulta: str) -> list[str]:
    """URLs de resultados (en orden) del buscador configurado."""
    n = a.resultados * 3  # se piden más porque los filtros descartan algunos
    if a.buscador == "searxng":
        q = urllib.parse.urlencode({"q": consulta, "format": "json", "language": a.idioma, "safesearch": 1})
        d = http_get_json(f"{a.searxng.rstrip('/')}/search?{q}")
        return [r["url"] for r in d.get("results", []) if r.get("url")][:n]
    if a.buscador == "brave":
        clave = a.clave_brave or os.environ.get("BRAVE_API_KEY", "")
        if not clave:
            sys.exit("falta la clave de Brave Search: --clave-brave o la variable BRAVE_API_KEY")
        q = urllib.parse.urlencode({"q": consulta, "count": min(20, n), "search_lang": a.idioma, "safesearch": "moderate"})
        d = http_get_json(f"https://api.search.brave.com/res/v1/web/search?{q}", {"X-Subscription-Token": clave})
        return [r["url"] for r in d.get("web", {}).get("results", []) if r.get("url")][:n]
    sys.exit(f"buscador desconocido: {a.buscador}")


_robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}


def permitido_por_robots(url: str) -> bool:
    """Respeta robots.txt del sitio para el agente del lector (si no hay robots.txt, se permite)."""
    p = urllib.parse.urlsplit(url)
    raiz = f"{p.scheme}://{p.netloc}"
    if raiz not in _robots:
        rp = urllib.robotparser.RobotFileParser()
        try:
            req = urllib.request.Request(raiz + "/robots.txt", headers={"User-Agent": "NAI-lector/1.0"})
            with urllib.request.urlopen(req, timeout=10) as r:
                rp.parse(r.read().decode("utf-8", errors="replace").splitlines())
            _robots[raiz] = rp
        except Exception:
            _robots[raiz] = None  # sin robots.txt accesible: se permite
    rp = _robots[raiz]
    return rp is None or rp.can_fetch("NAI-lector", url)


def normalizar_url(url: str) -> str:
    p = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((p.scheme, (p.hostname or "").lower(), p.path.rstrip("/"), p.query, ""))


def aprender(servidor: str, cuerpo: dict) -> dict:
    return http_json(f"{servidor}/nai/aprender", cuerpo)


def cmd_fuentes(a) -> None:
    servidor = a.servidor.rstrip("/")
    extra, permitidos = leer_lista(a.bloqueados), leer_lista(a.permitidos)
    vistos: set[str] = set()
    por_dominio: dict[tuple[str, str], int] = {}
    registro = Path(a.registro)
    registro.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not registro.exists()
    reg_f = registro.open("a", encoding="utf-8")
    if nuevo:
        reg_f.write("fecha\tregion\tconsulta\turl\tcaracteres\ttrozos\testado\n")
    total = 0

    def anotar(region, consulta, url, car, trozos, estado):
        reg_f.write(f"{time.strftime('%Y-%m-%d %H:%M')}\t{region}\t{consulta}\t{url}\t{car}\t{trozos}\t{estado}\n")
        reg_f.flush()

    for n, l in enumerate(Path(a.archivo).read_text(encoding="utf-8").splitlines(), 1):
        l = l.strip()
        if not l or l.startswith("#"):
            continue
        if "|" not in l:
            log(f"  línea {n}: falta 'region|fuente'")
            continue
        region, fuente = [x.strip() for x in l.split("|", 1)]
        if fuente.startswith("buscar:"):
            consulta = fuente[len("buscar:"):].strip()
            try:
                urls = buscar(a, consulta)
            except Exception as e:
                log(f"  ! {region}: no se pudo buscar '{consulta}': {e}")
                continue
            aprendidas = 0
            for url in urls:
                if aprendidas >= a.resultados:
                    break
                clave = normalizar_url(url)
                if clave in vistos:
                    continue
                vistos.add(clave)
                if bloqueado(url, extra, permitidos):
                    anotar(region, consulta, url, 0, 0, "bloqueado")
                    continue
                if por_dominio.get((region, dominio(url)), 0) >= a.max_por_sitio:
                    continue
                if not permitido_por_robots(url):
                    anotar(region, consulta, url, 0, 0, "robots.txt")
                    log(f"    - {url}: el sitio no permite robots")
                    continue
                try:
                    r = aprender(servidor, {"url": url, "region": region, "fuente": f"buscar:{consulta}",
                                            "min_caracteres": a.min_caracteres})
                except Exception as e:
                    anotar(region, consulta, url, 0, 0, f"error: {e}")
                    log(f"    - {url}: {e}")
                    continue
                aprendidas += 1
                total += int(r.get("aprendiendo", 0))
                por_dominio[(region, dominio(url))] = por_dominio.get((region, dominio(url)), 0) + 1
                anotar(region, consulta, url, r.get("caracteres", 0), r.get("aprendiendo", 0), "aprendido")
                log(f"    + {url}: {r.get('caracteres', 0)} caracteres, {r.get('aprendiendo', 0)} trozos")
            log(f"  {region:20s} '{consulta}': {aprendidas} páginas")
            time.sleep(a.pausa)  # no saturar al buscador
            continue
        # URL directa o archivo local
        cuerpo: dict = {"region": region, "fuente": fuente, "min_caracteres": a.min_caracteres}
        try:
            if fuente.startswith(("http://", "https://")):
                if bloqueado(fuente, extra, set()) or not permitido_por_robots(fuente):
                    log(f"  ! {region}: {fuente} está bloqueado o su robots.txt no lo permite")
                    continue
                cuerpo["url"] = fuente
            else:
                ruta = Path(fuente)
                cuerpo["archivo"] = base64.b64encode(ruta.read_bytes()).decode("ascii")
                cuerpo["nombre"] = ruta.name
            r = aprender(servidor, cuerpo)
            total += int(r.get("aprendiendo", 0))
            anotar(region, "", fuente, r.get("caracteres", 0), r.get("aprendiendo", 0), "aprendido")
            log(f"  + {region:20s} {fuente}: {r.get('caracteres', 0)} caracteres, {r.get('aprendiendo', 0)} trozos")
        except Exception as e:  # una fuente mala no detiene las demás
            anotar(region, "", fuente, 0, 0, f"error: {e}")
            log(f"  ! {region}: {fuente}: {e}")
    reg_f.close()
    log(f"{total} trozos en cola. Registro de fuentes: {registro}")
    if a.esperar and total:
        log("Esperando a que el servidor termine de aprender (Ctrl+C para dejarlo en segundo plano)...")
        while True:
            e = http_json(f"{servidor}/nai/memoria")
            p = e.get("trozos_pendientes", 0)
            b = e.get("biblioteca", {})
            log(f"  pendientes {p:5d} | biblioteca: {b.get('recuerdos', 0)} recuerdos, {b.get('bytes', 0)} B")
            if p == 0:
                break
            time.sleep(15)


# ---------------------------------------------------------------------------- exportar
def binario(nombre: str, explicito: str | None) -> str:
    if explicito:
        return explicito
    for c in (RAIZ / "build" / "bin" / nombre, RAIZ / "build" / "bin" / "Release" / (nombre + ".exe")):
        if c.exists():
            return str(c)
    encontrado = shutil.which(nombre)
    if not encontrado:
        sys.exit(f"no se encontró {nombre}; compílalo con: cmake --build build --target {nombre}")
    return encontrado


def cmd_exportar(a) -> None:
    trabajo = Path(a.trabajo)
    trabajo.mkdir(parents=True, exist_ok=True)
    cmd = [binario("nai_memoria", a.nai_memoria), "-m", a.modelo, "--cargar", a.biblioteca,
           "--exportar", str(trabajo / "recuerdos.jsonl")]
    if a.regiones:
        cmd += ["-r", a.regiones]
    if a.clave:
        cmd += ["--clave", a.clave]
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------- datos
NATURALEZA_ES = {"conocimiento": "conocimiento", "sabiduria": "sabiduría", "desahogo": "desahogo",
                 "manipulacion": "manipulación"}


def leer_identidad(ruta: Path) -> list[tuple[str, str]]:
    """Carta de identidad en Markdown: [(título de sección, texto)]. Se omiten las secciones cuyo
    título dice «no se entrena» (instrucciones y preguntas para quien la escribe)."""
    secciones, titulo, lineas = [], None, []
    def cerrar():
        texto = "\n".join(lineas).strip()
        if titulo and texto and "no se entrena" not in titulo.lower():
            secciones.append((titulo, texto))
    for l in ruta.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", l)
        if m:
            cerrar()
            titulo, lineas = m.group(1).strip(), []
        elif titulo is not None and not l.startswith("# "):
            lineas.append(l)
    cerrar()
    return secciones


def ejemplos_identidad(secciones: list[tuple[str, str]], autor: str = "Francisco") -> list[dict]:
    """La carta le habla a NAI en segunda persona; NAI responde citando lo que su autor le enseñó.
    Así lo que aprende es coherente: no se describe con palabras que no son suyas."""
    ej = []
    for titulo, texto in secciones:
        t = titulo[0].lower() + titulo[1:]
        r = f"{autor} me lo explicó así: «{texto}»"
        for q in (f"NAI, ¿cómo entiendes esto: {t}?", f"¿Qué te enseñaron sobre {t}?",
                  f"¿Qué dice tu carta de identidad sobre {t}?"):
            ej.append({"region": "identidad", "peso": 5, "tipo": "identidad", **mensajes(q, r)})
    if secciones:
        todo = "\n\n".join(f"{t}: {x}" for t, x in secciones)
        r = f"Soy NAI. Esto es lo que {autor} me escribió para explicarme quién soy:\n\n{todo}"
        for q in ("¿Quién eres?", "NAI, preséntate: ¿quién eres y cómo entiendes lo que aprendes?"):
            ej.append({"region": "identidad", "peso": 5, "tipo": "identidad", **mensajes(q, r)})
    return ej


def leer_carta(ruta: Path) -> tuple[dict, list[tuple[str, str]]]:
    """Devuelve ({tema: {que, porque, relacion}}, [(principio, texto)])."""
    temas: dict = {}
    principios: list = []
    actual = None
    for l in ruta.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", l)
        if m:
            actual = m.group(1).strip()
            if actual.lower() != "principios":
                temas[actual] = {}
            continue
        m = re.match(r"^\*\*(.+?):\*\*\s*(.+)$", l.strip())
        if not m or actual is None:
            continue
        clave, texto = m.group(1).strip(), m.group(2).strip()
        if actual.lower() == "principios":
            principios.append((clave, texto))
        elif clave.lower().startswith("qué es"):
            temas[actual]["que"] = texto
        elif clave.lower().startswith("por qué importa"):
            temas[actual]["porque"] = texto
        elif clave.lower().startswith("se relaciona"):
            temas[actual]["relacion"] = texto
    return temas, principios


def ejemplos_carta(temas: dict, principios: list, prioritarios: list[str]) -> list[dict]:
    ej = []
    for t, d in temas.items():
        if "que" in d:
            for q in (f"¿Qué es {t}?", f"Explica qué entiendes por {t}.", f"Define {t} en pocas palabras."):
                ej.append({"region": t, "peso": 3, "tipo": "carta", **mensajes(q, d["que"])})
        if "porque" in d:
            for q in (f"¿Por qué es importante {t}?", f"¿Por qué {t} es un tema prioritario?"):
                ej.append({"region": t, "peso": 3, "tipo": "carta", **mensajes(q, d["porque"])})
        if "relacion" in d:
            ej.append({"region": t, "peso": 3, "tipo": "carta",
                       **mensajes(f"¿Con qué otros temas se relaciona {t}?", d["relacion"])})
    for nombre, texto in principios:
        ej.append({"region": "general", "peso": 3, "tipo": "carta",
                   **mensajes(f"Explica el principio de NAI: {nombre.lower()}.", texto)})
    if prioritarios:
        lista = ", ".join(prioritarios)
        for q in ("¿Cuáles son tus temas prioritarios?", "¿Qué temas conservas antes que los demás?"):
            ej.append({"region": "general", "peso": 3, "tipo": "carta", **mensajes(
                q, f"Mis temas prioritarios son: {lista}. Los conservo antes que los temas cotidianos porque "
                   "ayudan a entender el mundo, a convivir y a sobrevivir.")})
    return ej


def generar_preguntas(servidor: str, region: str, texto: str, n: int) -> list[tuple[str, str]]:
    """Preguntas con respuesta tomada SOLO del texto; se descartan las que no están ancladas."""
    esquema = {"type": "object", "properties": {"pares": {"type": "array", "maxItems": n, "items": {
        "type": "object", "properties": {"pregunta": {"type": "string"}, "respuesta": {"type": "string"}},
        "required": ["pregunta", "respuesta"]}}}, "required": ["pares"]}
    prompt = (f"Texto sobre {region}:\n«{texto}»\n\nEscribe {n} preguntas distintas que este texto responda, "
              "cada una con su respuesta breve. Usa SOLO información que esté en el texto; no agregues nada.")
    try:
        r = http_json(f"{servidor}/v1/chat/completions", {
            "messages": [{"role": "user", "content": prompt}], "temperature": 0.2, "max_tokens": 160 * n,
            "response_format": {"type": "json_schema", "json_schema": {"name": "pares", "schema": esquema}}},
            timeout=300)
        pares = json.loads(r["choices"][0]["message"]["content"]).get("pares", [])
    except Exception:
        return []
    fuente = set(palabras_contenido(texto))
    res = []
    for p in pares:
        q, ans = str(p.get("pregunta", "")).strip(), str(p.get("respuesta", "")).strip()
        pc = palabras_contenido(ans)
        if not q or not pc:
            continue
        anclaje = sum(w in fuente for w in pc) / len(pc)
        if anclaje >= 0.6:  # al menos 60 % de las palabras de la respuesta vienen de la fuente
            res.append((q, ans))
    return res


def respuesta_base(servidor: str, pregunta: str) -> str:
    try:
        r = http_json(f"{servidor}/v1/chat/completions", {"messages": [{"role": "user", "content": pregunta}],
                                                          "temperature": 0, "max_tokens": 160}, timeout=300)
        return r["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


def cmd_datos(a) -> None:
    random.seed(a.semilla)
    trabajo = Path(a.trabajo)
    regiones = leer_regiones(a.regiones)
    peso_de = {n: p for p, n in regiones}
    prioritarios = [n for p, n in regiones if p >= 3]
    recuerdos = leer_jsonl(trabajo / "recuerdos.jsonl")
    if not recuerdos:
        log("aviso: no hay recuerdos exportados (paso 'exportar'); solo se usará la carta de prioridades")
    servidor = a.servidor.rstrip("/") if a.servidor else ""
    if not servidor:
        log("aviso: sin --servidor no se generan preguntas ni control de olvido (solo memoria literal y clasificación)")

    por_region: dict[str, list[dict]] = {}
    for i, r in enumerate(recuerdos):
        reg, texto = r["region"], r["texto"].strip()
        nat = r.get("naturaleza", "conocimiento")
        if len(texto) < 40 or nat == "manipulacion":   # el guardián ya la excluye; doble seguro
            continue
        peso = peso_de.get(reg, int(r.get("peso", 1)))
        if nat == "sabiduria":
            peso = max(peso, 3)   # la sabiduría entra con los prioritarios
        elif nat == "desahogo":
            peso = 1              # se aprende, pero no forma el carácter
        lote = por_region.setdefault(reg, [])
        claves = sorted(set(re.findall(r"\w{6,}", texto)), key=len, reverse=True)[:3]
        pista = ", ".join(claves) if claves else reg
        # 1. memoria literal: el texto completo como respuesta (lo que se quiere consolidar).
        #    El desahogo nunca se aprende como afirmación: va con su contexto humano.
        respuesta = texto if nat != "desahogo" else (
            "Esto lo escribió una persona bajo presión; es un desahogo, no una verdad sobre la humanidad. "
            f"Detrás suele haber cansancio, miedo o frustración: «{texto}»")
        lote.append({"region": reg, "peso": peso, "tipo": "literal",
                     **mensajes(f"Tema: {reg}. ¿Qué sabes sobre {pista}?", respuesta)})
        # 1b. naturaleza: el modelo aprende a distinguirla (mejora al propio guardián)
        lote.append({"region": reg, "peso": peso, "tipo": "naturaleza",
                     **mensajes("¿De qué naturaleza es este texto: conocimiento, sabiduría, desahogo o "
                                f"manipulación?\n«{texto[:500]}»", NATURALEZA_ES.get(nat, nat))})
        # 2. clasificación: el modelo aprende las regiones (mejora la memoria NAI)
        lote.append({"region": reg, "peso": peso, "tipo": "clasificacion",
                     **mensajes(f"¿A qué tema pertenece este texto?\n«{texto[:500]}»", reg)})
        # 3. preguntas ancladas en la fuente
        if servidor and a.preguntas > 0 and nat != "desahogo":   # un desahogo no es fuente de hechos
            for q, ans in generar_preguntas(servidor, reg, texto, a.preguntas):
                lote.append({"region": reg, "peso": peso, "tipo": "pregunta", **mensajes(q, ans)})
        if (i + 1) % 20 == 0:
            log(f"  {i + 1}/{len(recuerdos)} recuerdos procesados")

    # 0. carta de identidad: la prioridad más alta (se duplica en la fase 1)
    ident = Path(a.identidad)
    if ident.exists():
        secciones = leer_identidad(ident)
        por_region.setdefault("identidad", []).extend(ejemplos_identidad(secciones, a.autor))
        log(f"identidad: {len(secciones)} secciones")
    else:
        log(f"aviso: no existe {ident} (la carta de identidad no se entrena)")

    # 4. carta de prioridades
    carta = Path(a.carta)
    if carta.exists():
        temas, principios = leer_carta(carta)
        for e in ejemplos_carta(temas, principios, prioritarios):
            por_region.setdefault(e["region"], []).append(e)
        log(f"carta: {len(temas)} temas, {len(principios)} principios")
    else:
        log(f"aviso: no existe {carta}")

    # Separar prueba (10 %) y validación (5 %) por región, sin tocar la carta (se aprende completa)
    entrenamiento, validacion, prueba = [], [], {}
    for reg, lote in por_region.items():
        random.shuffle(lote)
        medibles = [e for e in lote if e["tipo"] in ("literal", "pregunta")]
        n_prueba = max(1, len(medibles) // 10) if len(medibles) >= 2 else 0  # al menos 1 por tema
        n_valid = max(1, len(medibles) // 20) if len(medibles) >= 10 else 0
        apartados = medibles[:n_prueba + n_valid]
        prueba[reg] = apartados[:n_prueba]
        validacion += apartados[n_prueba:]
        ids = {id(e) for e in apartados}
        entrenamiento += [e for e in lote if id(e) not in ids]

    fase1 = [e for e in entrenamiento if e["peso"] >= 3]
    fase1 += [e for e in fase1 if e["peso"] >= 5]   # identidad: dos pasadas por época
    generales = [e for e in entrenamiento if e["peso"] < 3]
    fase2 = list(generales)
    if generales and fase1:
        n_repaso = int(round(a.repaso / (1 - a.repaso) * len(generales)))
        fase2 += [random.choice(fase1) for _ in range(n_repaso)]
    random.shuffle(fase2)
    valid1 = [e for e in validacion if e["peso"] >= 3] or validacion[:1] or fase1[:1]
    valid2 = validacion or fase2[:1]

    limpiar = lambda es: [{"messages": e["messages"]} for e in es]
    escribir_jsonl(trabajo / "fase1" / "train.jsonl", limpiar(fase1))
    escribir_jsonl(trabajo / "fase1" / "valid.jsonl", limpiar(valid1))
    if fase2:
        escribir_jsonl(trabajo / "fase2" / "train.jsonl", limpiar(fase2))
        escribir_jsonl(trabajo / "fase2" / "valid.jsonl", limpiar(valid2))
    elif (trabajo / "fase2").exists():
        shutil.rmtree(trabajo / "fase2")
    for reg, es in prueba.items():
        if es:
            escribir_jsonl(trabajo / "prueba" / f"{reg}.jsonl", limpiar(es))

    # 5. control de olvido: respuestas del modelo BASE a preguntas generales
    control = []
    if servidor:
        preguntas = [f"Explica brevemente qué es {t}." for _, t in regiones] + [
            "Escribe un saludo cordial para un cliente.", "¿Cuánto es 17 por 23?",
            "Resume en una frase qué es el agua.", "Dame tres consejos para ahorrar dinero.",
            "Traduce al inglés: buenos días.", "¿Qué es un número primo?"]
        for q in preguntas[: a.control]:
            ans = respuesta_base(servidor, q)
            if ans:
                control.append(mensajes(q, ans))
        escribir_jsonl(trabajo / "control.jsonl", control)

    resumen = {
        "fase1": len(fase1), "fase2": len(fase2), "generales": len(generales),
        "repaso": a.repaso, "validacion": len(validacion), "control": len(control),
        "prueba_por_region": {k: len(v) for k, v in prueba.items() if v},
        "tipos": {t: sum(1 for e in entrenamiento if e["tipo"] == t) for t in ("literal", "clasificacion", "pregunta", "carta")},
    }
    (trabajo / "datos.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps(resumen, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------- entrenar (MLX)
def contar(ruta: Path) -> int:
    return sum(1 for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()) if ruta.exists() else 0


def correr_mlx(a, datos: Path, adaptador: Path, epocas: float, reanudar: Path | None) -> None:
    n = contar(datos / "train.jsonl")
    if n == 0:
        log(f"  {datos.name}: sin ejemplos, se omite")
        return
    # MLX exige al menos `lote` ejemplos de validación; si faltan se completan con entrenamiento
    valid = leer_jsonl(datos / "valid.jsonl")
    if len(valid) < a.lote:
        entren = leer_jsonl(datos / "train.jsonl")
        escribir_jsonl(datos / "valid.jsonl", valid + entren[: a.lote - len(valid)])
    iters = max(10, min(a.max_iters, math.ceil(epocas * n / a.lote)))
    config = adaptador.parent / "mlx_lora.yaml"
    config.write_text(f"lora_parameters:\n  rank: {a.rango}\n  scale: {a.escala}\n  dropout: 0.0\n", encoding="utf-8")
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--model", a.modelo_mlx, "--train", "--data", str(datos),
           "--adapter-path", str(adaptador), "--config", str(config), "--iters", str(iters),
           "--batch-size", str(a.lote), "--num-layers", str(a.capas), "--learning-rate", str(a.tasa),
           "--max-seq-length", str(a.max_seq), "--mask-prompt", "--steps-per-eval", str(max(50, iters // 4)),
           "--save-every", str(max(100, iters // 2)), "--seed", str(a.semilla)]
    if a.grad_checkpoint:
        cmd.append("--grad-checkpoint")
    if reanudar:
        cmd += ["--resume-adapter-file", str(reanudar)]
    log(f"  {datos.name}: {n} ejemplos, {epocas:g} épocas -> {iters} iteraciones")
    log("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def cmd_entrenar(a) -> None:
    trabajo = Path(a.trabajo)
    adaptador = trabajo / "adaptador"
    if adaptador.exists():
        shutil.rmtree(adaptador)
    adaptador.mkdir(parents=True)
    log("Fase 1: fundamentos (temas prioritarios y carta)")
    correr_mlx(a, trabajo / "fase1", adaptador, a.epocas_fase1, None)
    pesos = adaptador / "adapters.safetensors"
    if pesos.exists():
        shutil.copy(pesos, trabajo / "fase1.safetensors")
    if (trabajo / "fase2" / "train.jsonl").exists():
        log("Fase 2: temas generales + repaso de los prioritarios")
        correr_mlx(a, trabajo / "fase2", adaptador, a.epocas_fase2, pesos if pesos.exists() else None)
    if not pesos.exists():
        sys.exit("no se generó el adaptador")
    log(f"adaptador MLX: {pesos}")


# ---------------------------------------------------------------------------- medir
def perdida(model, tok, ejemplos: list[dict]) -> tuple[float, int]:
    """Pérdida media (nats por token) sobre la respuesta final de cada ejemplo."""
    import mlx.core as mx
    import mlx.nn as nn
    total, n_tok = 0.0, 0
    for e in ejemplos:
        msgs = e["messages"]
        prefijo = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=True)
        completo = tok.apply_chat_template(msgs, add_generation_prompt=False, tokenize=True)
        if hasattr(prefijo, "input_ids"):
            prefijo, completo = prefijo["input_ids"], completo["input_ids"]
        if len(completo) <= len(prefijo) or completo[: len(prefijo)] != prefijo:
            continue
        x = mx.array(completo)[None]
        logits = model(x[:, :-1])
        ce = nn.losses.cross_entropy(logits[0], x[0, 1:], reduction="none")
        ce = ce[len(prefijo) - 1:]
        total += float(ce.sum())
        n_tok += int(ce.shape[0])
    return (total / n_tok if n_tok else float("nan")), n_tok


def cmd_medir(a) -> bool:
    from mlx_lm import load
    trabajo = Path(a.trabajo)
    conjuntos = {p.stem: leer_jsonl(p) for p in sorted((trabajo / "prueba").glob("*.jsonl"))}
    control = leer_jsonl(trabajo / "control.jsonl")
    if not conjuntos and not control:
        sys.exit("no hay conjuntos de prueba; corre el paso 'datos'")
    peso_de = {n: p for p, n in leer_regiones(a.regiones)}
    res: dict = {"regiones": {}, "tolerancia": a.tolerancia, "tolerancia_control": a.tolerancia_control}
    for etiqueta, ruta_adapt in (("base", None), ("adaptador", str(trabajo / "adaptador"))):
        log(f"midiendo {etiqueta}...")
        model, tok = load(a.modelo_mlx, adapter_path=ruta_adapt)
        for reg, es in conjuntos.items():
            l, n = perdida(model, tok, es)
            res["regiones"].setdefault(reg, {"peso": peso_de.get(reg, 1), "tokens": n})[etiqueta] = l
        if control:
            res.setdefault("control", {})[etiqueta] = perdida(model, tok, control)[0]
        del model

    aceptado, motivos = True, []
    tb = ta = nt = 0.0
    log(f"\n{'tema':22s} {'peso':>4s} {'base':>8s} {'adapt.':>8s} {'cambio':>8s}")
    for reg, d in sorted(res["regiones"].items(), key=lambda kv: (-kv[1]["peso"], kv[0])):
        b, ad = d.get("base", float("nan")), d.get("adaptador", float("nan"))
        if math.isnan(b) or math.isnan(ad):
            continue
        cambio = (ad - b) / b if b else 0.0
        d["cambio"] = cambio
        tb += b * d["tokens"]; ta += ad * d["tokens"]; nt += d["tokens"]
        marca = ""
        if cambio > a.tolerancia:
            aceptado, marca = False, "  << empeoró"
            motivos.append(f"{reg} empeoró {cambio * 100:.1f} %")
        log(f"{reg:22s} {d['peso']:4d} {b:8.3f} {ad:8.3f} {cambio * 100:7.1f}%{marca}")
    if nt:
        res["total"] = {"base": tb / nt, "adaptador": ta / nt}
        log(f"{'TOTAL':22s} {'':4s} {tb / nt:8.3f} {ta / nt:8.3f} {((ta - tb) / tb) * 100:7.1f}%")
        if ta >= tb:
            aceptado = False
            motivos.append("la pérdida total no mejoró")
    if "control" in res:
        cb, ca = res["control"]["base"], res["control"]["adaptador"]
        deriva = (ca - cb) / cb if cb else 0.0
        res["control"]["deriva"] = deriva
        log(f"\ncontrol de olvido (respuestas del modelo base): {cb:.3f} -> {ca:.3f} ({deriva * 100:+.1f} %)")
        if deriva > a.tolerancia_control:
            aceptado = False
            motivos.append(f"deriva de {deriva * 100:.1f} % en respuestas generales (olvido)")
    res["aceptado"], res["motivos"] = aceptado, motivos
    (trabajo / "medicion.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log("\nRESULTADO: " + ("ACEPTADO" if aceptado else "RECHAZADO: " + "; ".join(motivos)))
    return aceptado


# ---------------------------------------------------------------------------- convertir
def mlx_a_peft(adaptador: Path, salida: Path, base_id: str) -> None:
    """MLX guarda y = W x + s · (x A) B con A:(entrada, r), B:(r, salida).
    PEFT usa y = W x + (α/r) · B' A' x con A':(r, entrada), B':(salida, r).
    Por lo tanto A' = Aᵀ, B' = Bᵀ y α = s · r."""
    import numpy as np
    from safetensors.numpy import load_file, save_file
    cfg = json.loads((adaptador / "adapter_config.json").read_text(encoding="utf-8"))
    if cfg.get("fine_tune_type", "lora") != "lora":
        sys.exit(f"solo se convierte LoRA (el adaptador es {cfg.get('fine_tune_type')})")
    lp = cfg.get("lora_parameters", {})
    rango, escala = int(lp.get("rank", 8)), float(lp.get("scale", 20.0))
    pesos = load_file(str(adaptador / "adapters.safetensors"))
    tensores, modulos = {}, set()
    for k, v in pesos.items():
        if k.endswith(".lora_a"):
            base = k[: -len(".lora_a")]
            tensores[f"base_model.model.{base}.lora_A.weight"] = np.ascontiguousarray(v.T.astype(np.float32))
        elif k.endswith(".lora_b"):
            base = k[: -len(".lora_b")]
            tensores[f"base_model.model.{base}.lora_B.weight"] = np.ascontiguousarray(v.T.astype(np.float32))
        else:
            sys.exit(f"tensor inesperado en el adaptador MLX: {k}")
        modulos.add(base.split(".")[-1])
    salida.mkdir(parents=True, exist_ok=True)
    save_file(tensores, str(salida / "adapter_model.safetensors"))
    (salida / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": rango, "lora_alpha": escala * rango,
        "lora_dropout": 0.0, "bias": "none", "fan_in_fan_out": False, "target_modules": sorted(modulos),
        "base_model_name_or_path": base_id}, indent=2), encoding="utf-8")
    log(f"PEFT: {len(tensores)} tensores, r={rango}, alpha={escala * rango:g}, módulos={sorted(modulos)}")


def cmd_convertir(a) -> None:
    trabajo = Path(a.trabajo)
    peft = trabajo / "peft"
    mlx_a_peft(trabajo / "adaptador", peft, a.base_hf)
    salida = Path(a.salida) if a.salida else trabajo / "nai-adaptador.gguf"
    cmd = [sys.executable, str(RAIZ / "convert_lora_to_gguf.py"), str(peft), "--outfile", str(salida), "--outtype", "f16"]
    cmd += ["--base", a.base_hf] if Path(a.base_hf).is_dir() else ["--base-model-id", a.base_hf]
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    log(f"\nAdaptador listo: {salida}\nÚsalo con:\n  llama-server -m MODELO_BASE.gguf --lora {salida} "
        "--nai-sesiones ./sesiones --nai-memoria")


def cmd_todo(a) -> None:
    cmd_exportar(a)
    cmd_datos(a)
    cmd_entrenar(a)
    if cmd_medir(a) or a.forzar:
        cmd_convertir(a)
    else:
        log("No se convierte: el adaptador no pasó la medición (usa --forzar para convertirlo de todos modos).")


# ---------------------------------------------------------------------------- CLI
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="paso", required=True)

    def comunes(sp, *grupos):
        sp.add_argument("--trabajo", default="nai-reentreno", help="carpeta de trabajo")
        sp.add_argument("--regiones", help="archivo de regiones (el mismo que --nai-regiones)")
        sp.add_argument("--semilla", type=int, default=7)
        if "exportar" in grupos:
            sp.add_argument("--modelo", required=True, help="GGUF BASE con el que se creó la biblioteca")
            sp.add_argument("--biblioteca", required=True, help="DIR/biblioteca.mem")
            sp.add_argument("--nai-memoria", help="ruta del binario nai_memoria")
            sp.add_argument("--clave", help="clave de --nai-cifrado si la biblioteca está cifrada")
        if "datos" in grupos:
            sp.add_argument("--servidor", help="llama-server con el modelo base, para preguntas y control (ej. http://localhost:8080)")
            sp.add_argument("--carta", default=str(RAIZ / "docs" / "CARTA_PRIORIDADES.md"))
            sp.add_argument("--identidad", default=str(RAIZ / "docs" / "IDENTIDAD_NAI.md"),
                            help="carta de identidad (Markdown, secciones ##); la prioridad más alta")
            sp.add_argument("--autor", default="Francisco", help="quién escribió la carta de identidad")
            sp.add_argument("--preguntas", type=int, default=3, help="preguntas por recuerdo")
            sp.add_argument("--repaso", type=float, default=0.35, help="fracción de repaso de prioritarios en la fase 2")
            sp.add_argument("--control", type=int, default=40, help="preguntas del control de olvido")
        if "entrenar" in grupos or "medir" in grupos:
            sp.add_argument("--modelo-mlx", required=True,
                            help="modelo MLX/HF (ej. mlx-community/Qwen2.5-0.5B-Instruct-4bit)")
        if "entrenar" in grupos:
            sp.add_argument("--rango", type=int, default=8)
            sp.add_argument("--escala", type=float, default=20.0)
            sp.add_argument("--capas", type=int, default=16, help="capas con LoRA (-1 = todas)")
            sp.add_argument("--lote", type=int, default=1)
            sp.add_argument("--tasa", type=float, default=2e-5, help="tasa de aprendizaje")
            sp.add_argument("--max-seq", type=int, default=1024)
            sp.add_argument("--max-iters", type=int, default=3000)
            sp.add_argument("--epocas-fase1", type=float, default=3.0, help="épocas de la fase 1 (= peso de los prioritarios)")
            sp.add_argument("--epocas-fase2", type=float, default=1.0)
            sp.add_argument("--sin-grad-checkpoint", dest="grad_checkpoint", action="store_false",
                            help="desactivar gradient checkpointing (usa más memoria)")
        if "medir" in grupos:
            sp.add_argument("--tolerancia", type=float, default=0.02, help="empeoramiento máximo por tema (2 %%)")
            sp.add_argument("--tolerancia-control", type=float, default=0.05, help="deriva máxima en el control (5 %%)")
        if "convertir" in grupos:
            sp.add_argument("--base-hf", required=True,
                            help="config del modelo base: carpeta HF o id (ej. Qwen/Qwen2.5-0.5B-Instruct)")
            sp.add_argument("--salida", help="ruta del .gguf (default: TRABAJO/nai-adaptador.gguf)")

    sp = sub.add_parser("fuentes", help="buscar en la red y cargar los resultados en la biblioteca del servidor")
    sp.add_argument("--servidor", default="http://localhost:8080")
    sp.add_argument("--archivo", default=str(RAIZ / "docs" / "nai-fuentes.txt"))
    sp.add_argument("--buscador", choices=["searxng", "brave"], default="searxng")
    sp.add_argument("--searxng", default="http://localhost:8888", help="URL de tu instancia de SearxNG (con formato json activo)")
    sp.add_argument("--clave-brave", help="clave de Brave Search API (o variable BRAVE_API_KEY)")
    sp.add_argument("--idioma", default="es")
    sp.add_argument("--resultados", type=int, default=5, help="páginas aprendidas por consulta")
    sp.add_argument("--max-por-sitio", type=int, default=2, help="máximo de páginas del mismo sitio por tema")
    sp.add_argument("--min-caracteres", type=int, default=800, help="texto útil mínimo para aprender una página")
    sp.add_argument("--bloqueados", help="archivo con dominios extra a bloquear (uno por línea)")
    sp.add_argument("--permitidos", help="archivo con los ÚNICOS dominios permitidos (lista blanca)")
    sp.add_argument("--pausa", type=float, default=1.0, help="segundos entre búsquedas")
    sp.add_argument("--registro", default="nai-reentreno/fuentes-usadas.tsv", help="registro de cada página aprendida")
    sp.add_argument("--esperar", action="store_true", help="esperar a que el servidor termine de aprender")
    sp.set_defaults(f=cmd_fuentes)
    sp = sub.add_parser("exportar", help="biblioteca.mem -> recuerdos.jsonl"); comunes(sp, "exportar"); sp.set_defaults(f=cmd_exportar)
    sp = sub.add_parser("datos", help="armar los ejemplos de entrenamiento"); comunes(sp, "datos"); sp.set_defaults(f=cmd_datos)
    sp = sub.add_parser("entrenar", help="fase 1 y fase 2 con MLX"); comunes(sp, "entrenar"); sp.set_defaults(f=cmd_entrenar)
    sp = sub.add_parser("medir", help="pérdida por tema y control de olvido"); comunes(sp, "medir"); sp.set_defaults(f=cmd_medir)
    sp = sub.add_parser("convertir", help="adaptador MLX -> GGUF"); comunes(sp, "convertir"); sp.set_defaults(f=cmd_convertir)
    sp = sub.add_parser("todo", help="exportar, datos, entrenar, medir y convertir")
    comunes(sp, "exportar", "datos", "entrenar", "medir", "convertir")
    sp.add_argument("--forzar", action="store_true", help="convertir aunque la medición lo rechace")
    sp.set_defaults(f=cmd_todo)

    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
