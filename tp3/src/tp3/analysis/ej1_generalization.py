"""
Ej. 1, parte B — análisis de las corridas guardadas por `tp3.experiments.ej1_generalization`.

Genera en `<out>/plots/`:
- generalization_curves.png  MSE de train y test por época (media ± desvío entre folds y semillas).
- fold_metrics.png           accuracy, precision, recall y F1 de cada fold (umbral 0.85), con el mejor marcado.
- fold_f1.png                F1 de test de cada (semilla, fold), evidencia del mejor conjunto de entrenamiento.
- threshold_metrics.png      las cuatro métricas sobre el test según el umbral de detección.
y `<out>/summary.csv` con las métricas por (semilla, fold).
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from tp3.functions.accuracy import accuracy, f1_score, precision, recall
from tp3.functions.classification.confusion import confusion_matrix
from tp3.functions.classification.fraud import DEFAULT_THRESHOLD, fraud_classification
from tp3.types.test_result import TestResult

parser = argparse.ArgumentParser(description="Ej1 parte B: análisis de generalización.")
parser.add_argument("-o", "--out", type=Path, default=Path("res/output/ej1_generalization"), help="Directorio de resultados")
parser.add_argument("--show", action="store_true", help="Además de guardar, mostrar los gráficos")

METRICS = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1_score}
THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.01), 2)


@dataclass(frozen=True)
class Run:
    seed: int
    fold: int
    lr: float
    train_mse: npt.NDArray[np.float64]
    test_mse: npt.NDArray[np.float64]
    test_loss: float
    test_idx: npt.NDArray[np.int64]
    predictions: npt.NDArray[np.float64]
    result: TestResult
    metrics: dict[str, float]


def load_runs(out: Path) -> list[Run]:
    runs = []
    for path in sorted((out / "runs").glob("*.npz")):
        data = np.load(path)
        meta = json.loads(str(data["meta"]))
        runs.append(Run(
            seed=meta["seed"], fold=meta["fold"], lr=meta["learning_rate"],
            train_mse=data["train_mse"], test_mse=data["test_mse"], test_loss=float(data["test_loss"]),
            test_idx=data["test_idx"], predictions=data["predictions"],
            result=meta["test_result"], metrics=meta["metrics"],
        ))
    if not runs:
        raise SystemExit(f"No hay corridas en {out / 'runs'}; correr primero tp3.experiments.ej1_generalization")
    return runs


def metrics_at(threshold: float, predictions: npt.NDArray[np.float64], flagged: npt.NDArray[np.float64]) -> dict[str, float]:
    result = confusion_matrix(fraud_classification(tuple(predictions), tuple(flagged), threshold))
    return {name: metric(result) for name, metric in METRICS.items()} # pyright: ignore[reportArgumentType]


def summarize(runs: list[Run]) -> pd.DataFrame:
    return pd.DataFrame([{
        "seed": r.seed,
        "fold": r.fold,
        "train_mse": r.train_mse[-1],
        "test_mse": r.test_mse[-1],
        "test_hse": r.test_loss,
        **r.metrics,
        **r.result,
    } for r in runs])


def best_run(runs: list[Run]) -> Run:
    """Best training set: highest test F1 (balances precision and recall on the rare class), ties by lowest test MSE."""
    return max(runs, key=lambda r: (r.metrics["f1"], -r.test_mse[-1]))


def plot_generalization_curves(runs: list[Run], plots: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, curves in (("train", [r.train_mse for r in runs]), ("test", [r.test_mse for r in runs])):
        curves = np.array(curves)
        mean, std = curves.mean(0), curves.std(0)
        epochs = np.arange(1, len(mean) + 1)
        ax.plot(epochs, mean, label=name)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("época")
    ax.set_ylabel("MSE")
    ax.set_title(f"Train vs. test ({len(runs)} corridas: folds × semillas, η = {runs[0].lr:g})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "generalization_curves.png", dpi=150)


def plot_fold_metrics(runs: list[Run], best: Run, plots: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = list(METRICS)
    values = [[r.metrics[n] for r in runs] for n in names]
    ax.boxplot(values, tick_labels=names, widths=0.5)
    rng = np.random.default_rng(0)
    for i, column in enumerate(values, start=1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(column)), column, s=12, alpha=0.6, color="tab:blue")
    ax.scatter(range(1, len(names) + 1), [best.metrics[n] for n in names], marker="*", s=180, color="tab:red",
               zorder=3, label=f"mejor: semilla {best.seed}, fold {best.fold}")
    ax.set_ylabel("valor sobre el fold de test")
    ax.set_title(f"Métricas por fold (umbral {DEFAULT_THRESHOLD})")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(plots / "fold_metrics.png", dpi=150)


def plot_fold_f1(runs: list[Run], best: Run, plots: Path) -> None:
    """Test F1 of every (seed, fold) as grouped bars: the evidence for picking the best training set."""
    seeds = sorted({r.seed for r in runs})
    folds = sorted({r.fold for r in runs})
    width = 0.8 / len(seeds)
    f1 = {(r.seed, r.fold): r.metrics["f1"] for r in runs}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, seed in enumerate(seeds):
        x = np.array(folds) + (i - (len(seeds) - 1) / 2) * width
        values = [f1[(seed, fold)] for fold in folds]
        bars = ax.bar(x, values, width, label=f"semilla {seed}")
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    best_x = best.fold + (seeds.index(best.seed) - (len(seeds) - 1) / 2) * width
    ax.scatter(best_x, best.metrics["f1"] + 0.012, marker="*", s=180, color="tab:red", zorder=3,
               label=f"mejor: semilla {best.seed}, fold {best.fold}")
    mean = np.mean(list(f1.values()))
    ax.axhline(mean, color="gray", ls=":", label=f"media ({mean:.3f})")
    values = list(f1.values())
    # Room on top for the legend, so it never covers a bar
    ax.set_ylim(min(values) - 0.03, max(values) + 0.06)
    ax.set_xticks(folds, [f"fold {f}" for f in folds])
    ax.set_ylabel("F1 sobre el fold de test")
    ax.set_title(f"F1 por fold (umbral {DEFAULT_THRESHOLD})")
    ax.legend(fontsize=8, loc="upper left", ncols=3)
    fig.tight_layout()
    fig.savefig(plots / "fold_f1.png", dpi=150)


def plot_threshold_metrics(sweep: dict[str, npt.NDArray[np.float64]], plots: Path) -> float:
    """Plots each metric vs. threshold (mean ± std over runs) and returns the threshold that maximizes mean F1."""
    best_threshold = float(THRESHOLDS[np.argmax(sweep["f1"].mean(0))])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, values in sweep.items():
        mean, std = values.mean(0), values.std(0)
        ax.plot(THRESHOLDS, mean, label=name)
        ax.fill_between(THRESHOLDS, mean - std, mean + std, alpha=0.2)
    ax.axvline(DEFAULT_THRESHOLD, color="gray", ls=":", label=f"umbral de BigModel ({DEFAULT_THRESHOLD})")
    ax.axvline(best_threshold, color="black", ls="--", label=f"máx. F1 ({best_threshold:.2f})")
    ax.set_xlabel("umbral de detección")
    ax.set_ylabel("valor sobre el fold de test")
    ax.set_title("Métricas según el umbral (media ± desvío entre corridas)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(plots / "threshold_metrics.png", dpi=150)
    return best_threshold


def main() -> None:
    args = parser.parse_args()
    runs = load_runs(args.out)
    reference = dict(np.load(args.out / "reference.npz"))
    flagged = reference["flagged"]

    summary = summarize(runs)
    summary.to_csv(args.out / "summary.csv", index=False)
    best = best_run(runs)

    sweep_rows = [[metrics_at(t, r.predictions, flagged[r.test_idx]) for t in THRESHOLDS] for r in runs]
    sweep = {name: np.array([[row[name] for row in run] for run in sweep_rows]) for name in METRICS}

    plots = args.out / "plots"
    plots.mkdir(exist_ok=True)
    plot_generalization_curves(runs, plots)
    plot_fold_metrics(runs, best, plots)
    plot_fold_f1(runs, best, plots)
    best_threshold = plot_threshold_metrics(sweep, plots)

    columns = ["train_mse", "test_mse", *METRICS]
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print()
    print("Media ± desvío entre corridas (umbral %.2f):" % DEFAULT_THRESHOLD)
    for column in columns:
        print(f"  {column:<10} {summary[column].mean():.4f} ± {summary[column].std():.4f}")
    print()
    print(f"Mejor conjunto de entrenamiento: semilla {best.seed}, fold {best.fold} "
          f"(F1 = {best.metrics['f1']:.4f}, test MSE = {best.test_mse[-1]:.5f})")
    at_best = {name: values[:, np.flatnonzero(THRESHOLDS == best_threshold)[0]].mean() for name, values in sweep.items()}
    print(f"Umbral con máx. F1 medio: {best_threshold:.2f} → " + ", ".join(f"{n}={v:.4f}" for n, v in at_best.items()))
    print(f"Gráficos en {plots}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
