#!/usr/bin/env python3
# [NAI] Borrador de auto-especulación por capas ("tronco de capas respirantes" sin pérdida)
#
# Crea un GGUF "borrador" con las primeras K capas del modelo original más su norma final y
# su cabeza de salida. Ese borrador es exactamente la salida temprana en la capa K que describe
# el README (nai_logits_early), pero se usa como modelo draft de la decodificación especulativa
# que llama.cpp ya trae:
#
#   1. el borrador (capas 1..K) propone varios tokens seguidos, barato;
#   2. el modelo completo (capas 1..N) los verifica todos en un solo paso;
#   3. se aceptan los tokens en los que ambos coinciden.
#
# Resultado: la salida es la del modelo completo (no se pierde calidad, a diferencia de un early
# exit directo) y se gana velocidad en la medida en que la capa K ya "resolvió" los tokens.
# Para elegir K, mide primero con tests/nai-manual/nai_medir_capas (regla Δh > τ del README).
#
# Uso:
#   python3 tools/nai/nai_borrador_capas.py modelo.gguf borrador.gguf --capas 12
#   llama-server -m modelo.gguf -md borrador.gguf --spec-draft-n-max 8 --spec-draft-n-min 1 [-ngld 99]
#
# Notas:
#   - Funciona con arquitecturas de bloques "blk.N.*" (llama, qwen2, qwen3, mistral, gemma...).
#   - Los metadatos por capa (arreglos de longitud N) se recortan a K.
#   - Se rechazan modelos con capas MTP/nextn (block_count incluye capas extra).
#   - Costo de memoria: duplica los pesos de las capas 1..K (aprox. K/N del modelo) y el embedding.
#   - Sin entrenamiento tipo LayerSkip, la aceptación con K pequeña puede ser baja: mide antes.

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
if (RAIZ / "gguf-py").is_dir():
    sys.path.insert(0, str(RAIZ / "gguf-py"))

import gguf  # noqa: E402

PATRON_BLOQUE = re.compile(r"^blk\.(\d+)\.")


def crear_borrador(entrada: Path, salida: Path, k: int, forzar: bool = False) -> dict:
    lector = gguf.GGUFReader(entrada, "r")

    campo_arch = lector.get_field(gguf.Keys.General.ARCHITECTURE)
    if campo_arch is None:
        raise ValueError("el GGUF no tiene general.architecture")
    arch = campo_arch.contents()

    clave_bloques = f"{arch}.block_count"
    campo_bloques = lector.get_field(clave_bloques)
    if campo_bloques is None:
        raise ValueError(f"el GGUF no tiene {clave_bloques}")
    n_capas = int(campo_bloques.contents())

    campo_nextn = lector.get_field(f"{arch}.nextn_predict_layers")
    if campo_nextn is not None and int(campo_nextn.contents()) > 0:
        raise ValueError("modelos con capas MTP/nextn no están soportados")

    if not 1 <= k < n_capas:
        raise ValueError(f"--capas debe estar entre 1 y {n_capas - 1} (el modelo tiene {n_capas})")

    if salida.exists() and not forzar:
        raise FileExistsError(f"{salida} ya existe (usa --forzar para sobrescribir)")

    escritor = gguf.GGUFWriter(salida, arch=arch, endianess=lector.endianess)

    recortados = []
    for campo in lector.fields.values():
        if campo.name == gguf.Keys.General.ARCHITECTURE or campo.name.startswith("GGUF."):
            continue  # los escribe GGUFWriter
        tipo = campo.types[0]
        subtipo = campo.types[-1] if tipo == gguf.GGUFValueType.ARRAY else None
        valor = campo.contents()

        if campo.name == clave_bloques:
            valor = k
        elif campo.name == gguf.Keys.General.NAME:
            valor = f"{valor} (NAI borrador {k}/{n_capas} capas)"
        elif (tipo == gguf.GGUFValueType.ARRAY and campo.name.startswith(f"{arch}.")
              and isinstance(valor, list) and len(valor) == n_capas):
            valor = valor[:k]  # metadato por capa
            recortados.append(campo.name)

        escritor.add_key_value(campo.name, valor, tipo, sub_type=subtipo)

    tensores = []
    for t in lector.tensors:
        m = PATRON_BLOQUE.match(t.name)
        if m and int(m.group(1)) >= k:
            continue
        tensores.append(t)

    for t in tensores:
        escritor.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

    escritor.write_header_to_file()
    escritor.write_kv_data_to_file()
    escritor.write_ti_data_to_file()
    for t in tensores:
        escritor.write_tensor_data(t.data, tensor_endianess=lector.endianess)
    escritor.close()

    bytes_orig = sum(t.n_bytes for t in lector.tensors)
    bytes_borr = sum(t.n_bytes for t in tensores)
    return {
        "arquitectura": arch,
        "capas_original": n_capas,
        "capas_borrador": k,
        "tensores": len(tensores),
        "metadatos_recortados": recortados,
        "mib_original": bytes_orig / 2**20,
        "mib_borrador": bytes_borr / 2**20,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="[NAI] Crea un borrador con las primeras K capas para auto-especulación")
    ap.add_argument("entrada", type=Path, help="modelo GGUF original")
    ap.add_argument("salida", type=Path, help="GGUF borrador a crear")
    ap.add_argument("--capas", "-k", type=int, required=True, help="número de capas a conservar (K)")
    ap.add_argument("--forzar", action="store_true", help="sobrescribir la salida si existe")
    args = ap.parse_args()

    try:
        info = crear_borrador(args.entrada, args.salida, args.capas, args.forzar)
    except (ValueError, FileExistsError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"[NAI] borrador creado: {args.salida}")
    print(f"  arquitectura   : {info['arquitectura']}")
    print(f"  capas          : {info['capas_borrador']} de {info['capas_original']}")
    print(f"  tamaño         : {info['mib_borrador']:.1f} MiB (original {info['mib_original']:.1f} MiB)")
    if info["metadatos_recortados"]:
        print(f"  metadatos por capa recortados: {', '.join(info['metadatos_recortados'])}")
    print("\nUso:")
    print(f"  llama-server -m {args.entrada} -md {args.salida} --spec-draft-n-max 8 --spec-draft-n-min 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
