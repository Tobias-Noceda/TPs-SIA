"""
Ej. 1, parte A — análisis de las corridas guardadas por `tp3.experiments.ej1_learning`.

Genera en `<out>/plots/`:
- loss_curves.png       loss por época, una curva por learning rate (media ± desvío entre semillas).
- best_comparison.png   mejor η de cada modelo superpuestos, con la referencia de predecir la media.
- pc1_curves.png        datos de BigModel y curva de cada modelo sobre PC1 (dirección de mayor varianza).
- distributions.png     histograma de predicciones vs. target.
- pred_vs_target.png    predicción vs. salida de BigModel (corrida mediana del mejor η).
y `<out>/summary.csv` con las métricas por (modelo, η).
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from tp3.config import load_config

parser = argparse.ArgumentParser(description="Ej1 parte A: análisis de aprendizaje.")
parser.add_argument("-o", "--out", type=Path, default=Path("res/output/ej1_learning"), help="Directorio de resultados")
parser.add_argument("--show", action="store_true", help="Además de guardar, mostrar los gráficos")

# Relative improvement under which the curve is considered flat (plateau / saturation).
PLATEAU_TOLERANCE = 1e-3
PLATEAU_WINDOW = 50
# A logistic unit is "saturated" on a sample when its slope is below this fraction of the max slope.
SATURATION_SLOPE = 0.1


@dataclass(frozen=True)
class Run:
    model: str
    activation: str
    lr: float
    seed: int
    seconds: float
    diverged: bool
    loss: npt.NDArray[np.float64]
    final_loss: float
    predictions: npt.NDArray[np.float64]
    h: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    bias: float
    config: str


def load_runs(out: Path) -> list[Run]:
    runs = []
    for path in sorted((out / "runs").glob("*.npz")):
        data = np.load(path)
        meta = json.loads(str(data["meta"]))
        runs.append(Run(
            model=meta["model"], activation=meta["activation"], lr=meta["learning_rate"], seed=meta["seed"],
            seconds=meta["train_seconds"], diverged=meta["diverged"], loss=data["loss"],
            final_loss=float(data["final_loss"]), predictions=data["predictions"], h=data["h"],
            weights=data["weights"], bias=float(data["bias"][0]), config=meta["config"],
        ))
    if not runs:
        raise SystemExit(f"No hay corridas en {out / 'runs'}; correr primero tp3.experiments.ej1_learning")
    return runs


def plateau_epoch(loss: npt.NDArray[np.float64]) -> int:
    """First epoch after which the loss improves less than PLATEAU_TOLERANCE (relative) over PLATEAU_WINDOW epochs."""
    for e in range(len(loss) - PLATEAU_WINDOW):
        if (loss[e] - loss[e + PLATEAU_WINDOW]) / loss[e] < PLATEAU_TOLERANCE:
            return e
    return len(loss)


def logistic_saturation(predictions: npt.NDArray[np.float64]) -> float:
    """Fraction of samples where logistic slope 2β·σ(1-σ) is below SATURATION_SLOPE of its max (β/2): 4σ(1-σ) < tol."""
    return float(np.mean(4 * predictions * (1 - predictions) < SATURATION_SLOPE))


def converged(runs: list[Run]) -> list[Run]:
    return [r for r in runs if not r.diverged]


def summarize(groups: dict[tuple[str, float], list[Run]], target: npt.NDArray[np.float64], reference: dict) -> pd.DataFrame:
    rows = []
    for (model, lr), all_runs in sorted(groups.items()):
        runs = converged(all_runs)
        row = {"model": model, "lr": lr, "seeds": len(all_runs), "diverged": len(all_runs) - len(runs)}
        if runs:
            # From the trained weights; r.loss is measured during the epoch, while still updating
            final = np.array([r.final_loss for r in runs])
            mse = np.array([np.mean((r.predictions - target) ** 2) for r in runs])
            mae = np.array([np.mean(np.abs(r.predictions - target)) for r in runs])
            out_of_range = np.array([np.mean((r.predictions < 0) | (r.predictions > 1)) for r in runs])
            nonlinear = runs[0].activation != "linear"
            row |= {
                "loss_final": final.mean(),
                "loss_final_std": final.std(),
                "mae": mae.mean(),
                "r2": 1 - mse.mean() / reference["mean_baseline_mse"],
                "plateau_epoch": int(np.median([plateau_epoch(r.loss) for r in runs])),
                "out_of_[0,1]": out_of_range.mean(),
                "saturated_samples": np.mean([logistic_saturation(r.predictions) for r in runs]) if nonlinear else np.nan,
                "seconds": np.mean([r.seconds for r in runs]),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def median_run(runs: list[Run]) -> Run:
    return sorted(converged(runs), key=lambda r: r.final_loss)[len(converged(runs)) // 2]


def loss_label(reference: dict) -> str:
    return str(reference["loss_function"]).upper()


def add_baseline(ax, reference: dict) -> None:
    ax.axhline(reference["mean_baseline_loss"], color="gray", ls=":", label="predecir la media")


def plot_mean_std(ax, runs: list[Run], label: str) -> None:
    curves = np.array([r.loss for r in converged(runs)])
    mean, std = curves.mean(0), curves.std(0)
    epochs = np.arange(1, len(mean) + 1)
    ax.plot(epochs, mean, label=label)
    ax.fill_between(epochs, mean - std, mean + std, alpha=0.2)


def plot_loss_curves(groups, models, reference, plots: Path) -> None:
    # Same y range on every panel so the final loss level of each model can be compared by eye
    curves = [r.loss for runs in groups.values() for r in converged(runs)]
    y_min = min(c.min() for c in curves) * 0.8
    y_max = max(max(c.max() for c in curves), float(reference["mean_baseline_loss"])) * 1.25

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.5), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        for (m, lr), runs in sorted(groups.items()):
            if m != model:
                continue
            if converged(runs):
                plot_mean_std(ax, runs, f"η = {lr:g}")
            else:
                ax.plot([], [], ls="none", label=f"η = {lr:g} (diverge)")
        add_baseline(ax, reference)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(y_min, y_max)
        ax.tick_params(labelleft=True)
        ax.set_title(model)
        ax.set_xlabel("época")
        ax.set_ylabel(loss_label(reference))
        ax.legend(fontsize=8)
    fig.suptitle("Curvas de aprendizaje (media ± desvío entre semillas)")
    fig.tight_layout()
    fig.savefig(plots / "loss_curves.png", dpi=150)


def plot_best_comparison(best: dict[str, list[Run]], reference, plots: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, runs in best.items():
        plot_mean_std(ax, runs, f"{model} (η = {runs[0].lr:g})")
    add_baseline(ax, reference)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("época")
    ax.set_ylabel(loss_label(reference))
    ax.set_title("Mejor learning rate de cada perceptrón")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "best_comparison.png", dpi=150)



def plot_pc1_curves(best, reference: dict, plots: Path) -> None:
    """
    Projects the 9-D problem onto PC1 (max variance direction of the standardized features).
    Each model is drawn as the curve it produces along that axis, f(t) = g(w · (μ + t·v) + b),
    so the points off the curve are the variance that the other 8 directions carry.
    """
    scores, target, pc1, mu = reference["pc1_scores"], reference["target"], reference["pc1"], reference["feature_mean"]
    t = np.linspace(scores.min(), scores.max(), 400)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(scores, target, s=6, alpha=0.25, color="lightgray", label="BigModel (target)")
    for model, runs in best.items():
        run = median_run(runs)
        activation, _ = load_config(run.config)["perceptron"]["activation_function"]  # type: ignore[misc]
        curve = activation(t * float(run.weights @ pc1) + float(run.weights @ mu) + run.bias)
        ax.plot(t, curve, lw=2.5, label=f"{model} (η = {run.lr:g})")
    for y in (0, 1):
        ax.axhline(y, color="gray", ls=":", lw=0.8)
    # The linear model keeps growing past 1; clip so the data and the logistic shape stay readable
    ax.set_ylim(-0.1, 1.25)
    ax.set_xlabel(f"PC1 — dirección de mayor varianza ({float(reference['pc1_explained_variance']):.0%} de la varianza)")
    ax.set_ylabel("probabilidad de fraude")
    ax.set_title("Curva de cada modelo sobre la componente principal")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(plots / "pc1_curves.png", dpi=150)


def plot_distributions(best, target, plots: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    predictions = [median_run(r).predictions for r in best.values()]
    bins = np.linspace(min(-0.3, *(p.min() for p in predictions)), max(1.3, *(p.max() for p in predictions)), 60)
    ax.hist(target, bins=bins, alpha=0.4, color="gray", label="BigModel (target)")
    for model, p in zip(best, predictions):
        ax.hist(p, bins=bins, histtype="step", lw=2, label=model)
    ax.axvline(0, color="red", lw=0.8)
    ax.axvline(1, color="red", lw=0.8)
    ax.set_xlabel("probabilidad de fraude")
    ax.set_ylabel("muestras")
    ax.set_title("Distribución de salidas")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "distributions.png", dpi=150)


def plot_pred_vs_target(best, target, flagged, plots: Path) -> None:
    fig, axes = plt.subplots(1, len(best), figsize=(5.5 * len(best), 5), squeeze=False)
    for ax, (model, runs) in zip(axes[0], best.items()):
        run = median_run(runs)
        r2 = 1 - np.mean((run.predictions - target) ** 2) / np.var(target)
        # Ground truth only colors the points, it is never used to train
        for label, color, name in ((0, "tab:blue", "legítima"), (1, "tab:red", "fraude (flagged)")):
            mask = flagged == label
            ax.scatter(target[mask], run.predictions[mask], s=3, alpha=0.3, color=color, label=name)
        ax.plot([0, 1], [0, 1], color="black", ls="--", lw=1)
        ax.axhspan(-1, 0, color="red", alpha=0.07)
        ax.axhspan(1, 2, color="red", alpha=0.07)
        ax.set_ylim(min(-0.05, run.predictions.min() - 0.05), max(1.05, run.predictions.max() + 0.05))
        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel("BigModel (target)")
        ax.set_ylabel("predicción")
        ax.set_title(f"{model} (η = {run.lr:g}, R² = {r2:.3f})")
        ax.legend(fontsize=8, markerscale=4)
    fig.suptitle("Predicción vs. target — zona roja: fuera de [0, 1], no es una probabilidad")
    fig.tight_layout()
    fig.savefig(plots / "pred_vs_target.png", dpi=150)



def main() -> None:
    args = parser.parse_args()
    runs = load_runs(args.out)
    reference = dict(np.load(args.out / "reference.npz"))
    target, flagged = reference["target"], reference["flagged"]

    groups: dict[tuple[str, float], list[Run]] = defaultdict(list)
    for run in runs:
        groups[(run.model, run.lr)].append(run)
    models = sorted({r.model for r in runs})

    summary = summarize(groups, target, reference)
    summary.to_csv(args.out / "summary.csv", index=False)

    valid = summary.dropna(subset=["loss_final"])
    best_lr = valid.loc[valid.groupby("model")["loss_final"].idxmin()].set_index("model")["lr"]
    best = {m: groups[(m, best_lr[m])] for m in models if m in best_lr}

    plots = args.out / "plots"
    plots.mkdir(exist_ok=True)
    plot_loss_curves(groups, models, reference, plots)
    plot_best_comparison(best, reference, plots)
    plot_pc1_curves(best, reference, plots)
    plot_distributions(best, target, plots)
    plot_pred_vs_target(best, target, flagged, plots)

    pd.set_option("display.width", 200)
    print(f"Muestras: {len(target)}")
    print(f"{loss_label(reference)} predecir la media: {float(reference['mean_baseline_loss']):.5g}")
    print(f"Varianza explicada por PC1:    {float(reference['pc1_explained_variance']):.1%}")
    print("PC1 (cargas por feature):      " + ", ".join(
        f"{c}={v:+.2f}" for c, v in zip(reference["input_columns"], reference["pc1"])))
    print()
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print()
    print("Mejor η por modelo:", ", ".join(f"{m} η={best_lr[m]:g}" for m in best))
    print(f"Gráficos en {plots}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
