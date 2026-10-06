#!/usr/bin/env bash
# Ej. 3: 784-256-10 con oculta logística, entrenando también con more_digits.csv, para llegar a accuracy >= 98%.
#   1. Estudios de res/config/ej3/studies.json (conjunto de entrenamiento, η × optimizador, técnicas), cada variante con
#      varias semillas; reusa el experimento del ej. 2 (cada corrida se evalúa en digits_test.csv).
#   2. Análisis: tablas + gráficos en res/output/ej3_digits/.
#
# Uso: ./run3.sh [estudios...]   (sin argumentos corre todos)
set -euo pipefail
cd "$(dirname "$0")"

OUT="res/output/ej3_digits"

ONLY=()
if [ $# -gt 0 ]; then
    ONLY=(--only "$@")
fi

# data*.npz (datasets parseados) se conservan: parsear los CSV tarda ~30 s
rm -rf "$OUT/runs" "$OUT/plots"

uv run --package tp3 ej2-digits -s res/config/ej3/studies.json "${ONLY[@]}" -o "$OUT"
MPLBACKEND=Agg uv run --package tp3 ej3-digits-analysis -o "$OUT"
