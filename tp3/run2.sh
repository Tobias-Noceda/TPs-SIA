#!/usr/bin/env bash
# Ej. 2: perceptrón multicapa para dígitos 0-9.
#   1. Estudios de res/config/ej2/studies.json (η × optimizador, arquitectura, activación oculta), cada
#      variante con varias semillas; cada semilla parte digits.csv 80/20 en train / validación.
#   2. Final: la configuración con mejor validación se reentrena con todo digits.csv y se evalúa en
#      digits_test.csv (único uso del test).
#   3. Análisis: tablas + gráficos en res/output/ej2_digits/.
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
uv run --package tp3 ej2-digits --final -o "$OUT"
MPLBACKEND=Agg uv run --package tp3 ej2-digits-analysis -o "$OUT"
