#!/usr/bin/env bash
# Ej. 2: perceptrón multicapa para dígitos 0-9.
#   1. Estudios de res/config/ej2/studies.json (η × optimizador incl. η adaptativo, arquitectura, activación oculta), cada
#      variante con varias semillas; cada corrida entrena con todo digits.csv y se evalúa en digits_test.csv.
#   2. Análisis: tablas + gráficos en res/output/ej2_digits/.
#
# Uso: ./run2.sh [estudios...]   (sin argumentos corre todos)
set -euo pipefail
cd "$(dirname "$0")"

OUT="res/output/ej2_digits"

ONLY=()
if [ $# -gt 0 ]; then
    ONLY=(--only "$@")
fi

# data.npz (datasets parseados) se conserva: parsear los CSV tarda ~10 s
rm -rf "$OUT/runs" "$OUT/final" "$OUT/plots"

uv run --package tp3 ej2-digits "${ONLY[@]}" -o "$OUT"
MPLBACKEND=Agg uv run --package tp3 ej2-digits-analysis -o "$OUT"
