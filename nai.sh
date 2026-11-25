#!/usr/bin/env bash
# [NAI] Arranca NAI completo: ./nai.sh  (opciones: ./nai.sh --help)
exec bash "$(cd "$(dirname "$0")" && pwd)/tools/nai/nai-arrancar.sh" "$@"
