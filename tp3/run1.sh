#!/usr/bin/env bash
# Ej. 1, parte A: aprendizaje del perceptrón simple lineal vs. no lineal (logística)
# sobre TODO fraud_dataset.csv. Primero corre los experimentos (guardan datos crudos),
# después el análisis (tabla + gráficos en res/output/ej1_learning/).
#
# Uso: ./run1.sh [épocas] [semillas] [batch_size]
set -euo pipefail
cd "$(dirname "$0")"

EPOCHS="${1:-1000}"
SEEDS="${2:-5}"
BATCH="${3:-32}"
OUT="res/output/ej1_learning"

rm -rf "$OUT"

# Rangos de η distintos por modelo: la logística tiene pendiente máxima 2β·¼ = 0.5,
# así que tolera η más grandes que la lineal antes de diverger.
uv run --package tp3 ej1-learning -c res/config/ej1/linear.json \
    --lrs 0.0001 0.001 0.01 0.1 --epochs "$EPOCHS" --seeds "$SEEDS" --batch-size "$BATCH" -o "$OUT"

uv run --package tp3 ej1-learning -c res/config/ej1/logistic.json \
    --lrs 0.001 0.01 0.1 1 --epochs "$EPOCHS" --seeds "$SEEDS" --batch-size "$BATCH" -o "$OUT"

MPLBACKEND=Agg uv run --package tp3 ej1-learning-analysis -o "$OUT"
