#!/usr/bin/env python3
# [NAI] Crea un modelo llama DIMINUTO con pesos aleatorios para probar el motor NAI sin
# descargar nada (CI, pruebas de humo). Usa el vocabulario de models/ggml-vocab-llama-spm.gguf.
# El texto que genera no tiene sentido: sirve para validar el código, no la calidad.
#
#   python3 tests/nai-manual/crear_modelo_prueba.py /tmp/nai-prueba.gguf [--capas 8]
#
# Con la forma de Qwen2.5-0.5B (vocabulario de 152k, para medir costos realistas por token):
#   python3 tests/nai-manual/crear_modelo_prueba.py /tmp/qwen05-forma.gguf --vocab models/ggml-vocab-qwen2.gguf \
#       --capas 24 --embd 896 --cabezas 14 --cabezas-kv 2 --ff 4864 --f16

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "gguf-py"))

import gguf  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("salida", type=Path)
    ap.add_argument("--capas", type=int, default=8)
    ap.add_argument("--embd", type=int, default=64)
    ap.add_argument("--cabezas", type=int, default=4)
    ap.add_argument("--cabezas-kv", type=int, default=0, help="0 = igual que --cabezas (sin GQA)")
    ap.add_argument("--f16", action="store_true", help="pesos en F16 (la mitad de tamaño)")
    ap.add_argument("--ff", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--semilla", type=int, default=1234)
    ap.add_argument("--vocab", type=Path, default=RAIZ / "models" / "ggml-vocab-llama-spm.gguf")
    a = ap.parse_args()

    vocab = gguf.GGUFReader(a.vocab, "r")
    w = gguf.GGUFWriter(a.salida, arch="llama")
    w.add_name("NAI modelo de prueba (aleatorio)")
    w.add_context_length(a.ctx)
    w.add_embedding_length(a.embd)
    w.add_block_count(a.capas)
    w.add_feed_forward_length(a.ff)
    w.add_head_count(a.cabezas)
    kv = a.cabezas_kv or a.cabezas
    w.add_head_count_kv(kv)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_dimension_count(a.embd // a.cabezas)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16 if a.f16 else gguf.LlamaFileType.ALL_F32)

    n_vocab = 0
    for campo in vocab.fields.values():
        if not campo.name.startswith("tokenizer."):
            continue
        tipo = campo.types[0]
        sub = campo.types[-1] if tipo == gguf.GGUFValueType.ARRAY else None
        valor = campo.contents()
        if campo.name == "tokenizer.ggml.tokens":
            n_vocab = len(valor)
        w.add_key_value(campo.name, valor, tipo, sub_type=sub)
    if n_vocab == 0:
        print("el GGUF de vocabulario no tiene tokens", file=sys.stderr)
        return 1

    rng = np.random.default_rng(a.semilla)

    def mat(filas: int, cols: int, escala: float) -> np.ndarray:
        m = (rng.standard_normal((filas, cols), dtype=np.float32) * escala)
        return m.astype(np.float16) if a.f16 else m

    e, ff = a.embd, a.ff
    e_kv = e // a.cabezas * kv
    w.add_tensor("token_embd.weight", mat(n_vocab, e, 0.5))
    for i in range(a.capas):
        w.add_tensor(f"blk.{i}.attn_norm.weight", np.ones(e, dtype=np.float32))
        w.add_tensor(f"blk.{i}.attn_q.weight", mat(e, e, e ** -0.5))
        w.add_tensor(f"blk.{i}.attn_k.weight", mat(e_kv, e, e ** -0.5))
        w.add_tensor(f"blk.{i}.attn_v.weight", mat(e_kv, e, e ** -0.5))
        w.add_tensor(f"blk.{i}.attn_output.weight", mat(e, e, 0.2 * e ** -0.5))
        w.add_tensor(f"blk.{i}.ffn_norm.weight", np.ones(e, dtype=np.float32))
        w.add_tensor(f"blk.{i}.ffn_gate.weight", mat(ff, e, e ** -0.5))
        w.add_tensor(f"blk.{i}.ffn_up.weight", mat(ff, e, e ** -0.5))
        w.add_tensor(f"blk.{i}.ffn_down.weight", mat(e, ff, 0.2 * ff ** -0.5))
    w.add_tensor("output_norm.weight", np.ones(e, dtype=np.float32))
    w.add_tensor("output.weight", mat(n_vocab, e, 1.0))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"[NAI] modelo de prueba: {a.salida} ({a.capas} capas, n_embd {e}, vocab {n_vocab})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
