#!/usr/bin/env bash
# Ej. 1, parte B: generalización del perceptrón logístico con k-fold cross validation
# estratificado. El z-score se ajusta con el train de cada fold y se reaplica a su test.
# Primero corre los experimentos (guardan datos crudos), después el análisis
# (tabla + gráficos en res/output/ej1_generalization/).
#
# Uso: ./run1_generalization.sh [épocas] [semillas] [folds] [η] [batch_size]
set -euo pipefail
cd "$(dirname "$0")"

EPOCHS="${1:-1000}"
SEEDS="${2:-3}"
FOLDS="${3:-5}"
LR="${4:-0.01}"
BATCH="${5:-32}"
OUT="res/output/ej1_generalization"

rm -rf "$OUT"

uv run --package tp3 ej1-generalization -c res/config/ej1/logistic.json \
    -k "$FOLDS" --seeds "$SEEDS" --lr "$LR" --epochs "$EPOCHS" --batch-size "$BATCH" -o "$OUT"

MPLBACKEND=Agg uv run --package tp3 ej1-generalization-analysis -o "$OUT"
