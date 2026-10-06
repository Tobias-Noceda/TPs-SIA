"""
Ej. 2 — análisis de las corridas guardadas por `tp3.experiments.ej2_digits`.

Genera en `<out>/plots/`:
- lr_optimizer.png     accuracy de validación final según η, una curva por optimizador.
- optimizer_curves.png pérdida de train y accuracy de validación por época, cada optimizador con su mejor η.
- architecture.png     accuracy de train vs. validación por arquitectura (capacidad y sobreajuste).
- hidden_activation.png accuracy de validación por semilla según la activación oculta, marcando el
                       dígito que la red dejó de predecir.
- confusion_test.png   matriz de confusión en digits_test.csv del modelo final (normalizada por fila).
y `<out>/summary.csv` con la accuracy de validación por configuración.

El modelo se elige SOLO por validación (particiones de digits.csv); el test se reporta al final.
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

parser = argparse.ArgumentParser(description="Ej2: análisis del estudio de dígitos.")
parser.add_argument("-o", "--out", type=Path, default=Path("res/output/ej2_digits"), help="Directorio de resultados")

CLASSES = 10
# A digit with a validation recall below this is considered lost (the network never predicts it)
LOST_RECALL = 0.1
# A variant whose mean validation accuracy is below this failed (did not learn or was unstable) in some seed
FAILED_ACCURACY = 0.9

# Categorical palette (fixed order) and sequential ramp for the confusion matrix
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MUTED = "#8a8984"
OPTIMIZERS = {"gradient_descent": "gradiente descendente", "momentum": "momentum (α = 0.9)", "adam": "Adam"}


@dataclass(frozen=True)
class Run:
    config: dict
    seed: int
    diverged: bool
    history: dict[str, npt.NDArray[np.float64]]
    predictions: npt.NDArray[np.int64]
    labels: npt.NDArray[np.int64]

    @property
    def final_validation(self) -> float:
        return 0.0 if self.diverged else float(self.history["validation_accuracy"][-1])


def load_runs(out: Path) -> dict[str, list[Run]]:
    runs: dict[str, list[Run]] = defaultdict(list)
    for path in sorted((out / "runs").glob("*.npz")):
        data = np.load(path)
        meta = json.loads(str(data["meta"]))
        history = {k: data[k] for k in ("train_loss", "train_accuracy", "validation_accuracy")}
        runs[meta["config_id"]].append(Run(meta["config"], meta["seed"], meta["diverged"], history, data["validation_predictions"], data["validation_labels"]))
    if not runs:
        raise SystemExit(f"No hay corridas en {out / 'runs'}; correr primero tp3.experiments.ej2_digits")
    return runs


def recall_per_class(labels: npt.NDArray[np.int64], predictions: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    """Fraction of each digit's samples predicted as that digit (NaN for digits absent from labels)."""
    return np.array([np.mean(predictions[labels == d] == d) if np.any(labels == d) else np.nan for d in range(CLASSES)])


def mean_std(values: list[float]) -> str:
    return f"{np.mean(values):.4f} ± {np.std(values):.4f}"


def summary_table(runs: dict[str, list[Run]], studies: dict) -> pd.DataFrame:
    rows = []
    for name, study in studies.items():
        for variant in study["variants"]:
            group = runs[variant["config_id"]]
            accuracies = [r.final_validation for r in group]
            train = [r.history["train_accuracy"][-1] for r in group if not r.diverged]
            rows.append({
                "study": name, "variant": variant["label"], "config_id": variant["config_id"],
                "optimizer": group[0].config["optimizer"]["name"], "learning_rate": group[0].config["learning_rate"],
                "seeds": len(group), "diverged": sum(r.diverged for r in group),
                "train_accuracy": np.mean(train) if train else np.nan,
                "validation_accuracy": np.mean(accuracies), "validation_std": np.std(accuracies),
            })
    return pd.DataFrame(rows)


def plot_lr_optimizer(table: pd.DataFrame, plots: Path) -> dict[str, str]:
    """Final validation accuracy vs. η per optimizer. Returns the config of the best η per optimizer."""
    study = table[table["study"] == "optimizer"]
    best: dict[str, str] = {}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Zoom on the useful range: variants where some seed did not learn (diverged or stuck at chance
    # level) are drawn on the bottom edge with their real mean
    learned = study["validation_accuracy"] >= FAILED_ACCURACY
    bottom = study.loc[learned, "validation_accuracy"].min() - 0.02
    for color, (optimizer, label) in zip(SERIES, OPTIMIZERS.items()):
        rows = study[study["optimizer"] == optimizer].sort_values("learning_rate")
        if rows.empty:
            continue
        shown = rows["validation_accuracy"].clip(lower=bottom)
        ax.plot(rows["learning_rate"], shown, color=color, marker="o", markersize=6, linewidth=2, label=label)
        ok = rows["validation_accuracy"] >= FAILED_ACCURACY
        ax.errorbar(rows["learning_rate"][ok], shown[ok], yerr=rows["validation_std"][ok], color=color, linestyle="none", capsize=3)
        for _, row in rows[~ok].iterrows():
            ax.annotate(f"falla\n(media {row['validation_accuracy']:.2f})", (row["learning_rate"], bottom), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
        best[optimizer] = rows.loc[rows["validation_accuracy"].idxmax(), "config_id"]
    ax.set_xscale("log")
    ax.set_xlabel("tasa de aprendizaje η")
    ax.set_ylabel("accuracy de validación (época final)")
    ax.set_ylim(bottom - 0.005, 1.0)
    ax.set_title("η × optimizador (media ± desvío entre semillas)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(plots / "lr_optimizer.png", dpi=150)
    plt.close(fig)
    return best


def plot_optimizer_curves(runs: dict[str, list[Run]], best: dict[str, str], plots: Path) -> None:
    fig, (loss_ax, accuracy_ax) = plt.subplots(1, 2, figsize=(11, 4.2))
    for color, (optimizer, label) in zip(SERIES, OPTIMIZERS.items()):
        if optimizer not in best:
            continue
        group = runs[best[optimizer]]
        lr = group[0].config["learning_rate"]
        for ax, key in ((loss_ax, "train_loss"), (accuracy_ax, "validation_accuracy")):
            values = np.array([r.history[key] for r in group])
            epochs = np.arange(1, values.shape[1] + 1)
            mean = np.nanmean(values, axis=0)
            ax.plot(epochs, mean, color=color, linewidth=2, label=f"{label}, η = {lr:g}")
            ax.fill_between(epochs, np.nanmin(values, axis=0), np.nanmax(values, axis=0), color=color, alpha=0.15, linewidth=0)
    loss_ax.set_yscale("log")
    loss_ax.set_ylabel("error cuadrático ½Σ (train)")
    accuracy_ax.set_ylabel("accuracy de validación")
    accuracy_ax.set_ylim(0.85, 1.0)
    for ax in (loss_ax, accuracy_ax):
        ax.set_xlabel("época")
        ax.grid(alpha=0.3)
    accuracy_ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Convergencia de cada optimizador con su mejor η (media; banda = mín.–máx. entre semillas)")
    fig.tight_layout()
    fig.savefig(plots / "optimizer_curves.png", dpi=150)
    plt.close(fig)


def plot_architecture(table: pd.DataFrame, runs: dict[str, list[Run]], plots: Path) -> None:
    study = table[table["study"] == "architecture"].reset_index(drop=True)
    labels = []
    for _, row in study.iterrows():
        sizes = [784, *runs[row["config_id"]][0].config["hidden"], CLASSES]
        parameters = sum(a * b + b for a, b in zip(sizes, sizes[1:]))
        labels.append(f"{row['variant']}\n{parameters / 1000:.0f}k pesos")
    x = np.arange(len(study))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.vlines(x, study["validation_accuracy"], study["train_accuracy"], color=MUTED, linewidth=1.5, zorder=1)
    ax.scatter(x, study["train_accuracy"], color=SERIES[1], s=50, zorder=2, label="train")
    ax.errorbar(x, study["validation_accuracy"], yerr=study["validation_std"], fmt="o", color=SERIES[0], markersize=7,
                capsize=3, zorder=3, label="validación")
    for i, row in study.iterrows():
        ax.annotate(f"{row['train_accuracy']:.3f}", (i, row["train_accuracy"]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
        ax.annotate(f"{row['validation_accuracy']:.3f}", (i, row["validation_accuracy"]), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=8)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(study["validation_accuracy"].min() - 0.008, 1.008)
    ax.set_ylabel("accuracy (época final)")
    ax.set_title("Arquitectura: train vs. validación (la brecha es el sobreajuste)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(plots / "architecture.png", dpi=150)
    plt.close(fig)


def plot_hidden_activation(runs: dict[str, list[Run]], studies: dict, plots: Path) -> None:
    variants = studies["hidden_activation"]["variants"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rng = np.random.default_rng(0)
    for i, (color, variant) in enumerate(zip(SERIES, variants)):
        group = runs[variant["config_id"]]
        accuracies = np.array([r.final_validation for r in group])
        lost = [np.flatnonzero(recall_per_class(r.labels, r.predictions) < LOST_RECALL) for r in group]
        has_lost = np.array([len(l) > 0 for l in lost])
        x = i + rng.uniform(-0.08, 0.08, len(group))
        ax.scatter(x[~has_lost], accuracies[~has_lost], color=color, s=45, zorder=2)
        ax.scatter(x[has_lost], accuracies[has_lost], facecolors="none", edgecolors=color, linewidths=1.5, s=45, zorder=2)
        ax.hlines(np.median(accuracies), i - 0.25, i + 0.25, color=color, linewidth=2, zorder=1)
        digits = sorted(int(d) for l in lost for d in l)
        summary = f"pierde un dígito: {has_lost.sum()}/{len(group)}" + (f"\n(dígitos {', '.join(map(str, digits))})" if digits else "")
        ax.annotate(summary, (i, 0), xycoords=("data", "axes fraction"), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    ax.set_xticks(range(len(variants)), [v["label"] for v in variants])
    ax.set_xlim(-0.6, len(variants) - 0.4)
    # Room at the bottom for the per-variant summary
    every = [r.final_validation for v in variants for r in runs[v["config_id"]]]
    ax.set_ylim(min(every) - 0.035, max(every) + 0.008)
    ax.set_ylabel("accuracy de validación (época final)")
    ax.set_title("Activación oculta, una semilla por punto\n(hueco = nunca predice algún dígito)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plots / "hidden_activation.png", dpi=150)
    plt.close(fig)


def final_report(out: Path, plots: Path) -> None:
    paths = sorted((out / "final").glob("s*.npz"))
    if not paths:
        print("\nSin corridas finales (correr ej2-digits --final).")
        return
    with open(out / "final" / "chosen.json") as f:
        chosen = json.load(f)
    finals = [np.load(p) for p in paths]
    labels = finals[0]["test_labels"]
    accuracies = [float(f["test_accuracy"][-1]) for f in finals]
    without_eights = [float(np.mean(f["test_predictions"][labels != 8] == labels[labels != 8])) for f in finals]
    recalls = np.array([recall_per_class(labels, f["test_predictions"]) for f in finals])

    print("\n== Modelo final (reentrenado con todo digits.csv) ==")
    print(json.dumps(chosen["config"], ensure_ascii=False))
    print(f"accuracy de validación (selección): {chosen['validation_accuracy']:.4f}")
    print(f"accuracy en test:                   {mean_std(accuracies)}  ({len(finals)} semillas)")
    print(f"accuracy en test sin los 8:         {mean_std(without_eights)}")
    print(f"techo en test sin predecir 8s:      {np.mean(labels != 8):.4f}")
    print("recall por dígito en test: " + "  ".join(f"{d}: {r:.3f}" for d, r in enumerate(recalls.mean(axis=0))))

    # Summed over seeds, normalized per true digit
    matrix = np.zeros((CLASSES, CLASSES))
    for f in finals:
        np.add.at(matrix, (labels, f["test_predictions"]), 1)
    matrix /= matrix.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    for i in range(CLASSES):
        for j in range(CLASSES):
            if matrix[i, j] >= 0.005:
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if matrix[i, j] > 0.5 else "black")
    ax.set_xticks(range(CLASSES))
    ax.set_yticks(range(CLASSES))
    ax.set_xlabel("dígito predicho")
    ax.set_ylabel("dígito real")
    ax.set_title(f"Test: accuracy {np.mean(accuracies):.3f} (fracción por fila, {len(finals)} semillas)")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(plots / "confusion_test.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parser.parse_args()
    plots = args.out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.out)
    with open(args.out / "studies.json") as f:
        studies = json.load(f)["studies"]

    data = np.load(args.out / "data.npz")
    train_counts = np.bincount(data["y"].argmax(1), minlength=CLASSES)
    test_counts = np.bincount(data["y_test"].argmax(1), minlength=CLASSES)
    print("muestras por dígito  digits.csv: " + " ".join(map(str, train_counts)))
    print("                digits_test.csv: " + " ".join(map(str, test_counts)))

    table = summary_table(runs, studies)
    table.to_csv(args.out / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    for name, study in table.groupby("study", sort=False):
        print(f"\n== {studies[name]['title']} ==")
        print(study[["variant", "optimizer", "seeds", "diverged", "train_accuracy", "validation_accuracy", "validation_std"]].to_string(index=False, float_format="%.4f"))

    if "optimizer" in studies:
        best = plot_lr_optimizer(table, plots)
        plot_optimizer_curves(runs, best, plots)
    if "architecture" in studies:
        plot_architecture(table, runs, plots)
    if "hidden_activation" in studies:
        plot_hidden_activation(runs, studies, plots)
        for variant in studies["hidden_activation"]["variants"]:
            lost = sum(np.any(recall_per_class(r.labels, r.predictions) < LOST_RECALL) for r in runs[variant["config_id"]])
            print(f"{variant['label']}: {lost}/{len(runs[variant['config_id']])} semillas pierden un dígito")

    final_report(args.out, plots)
    print(f"\nGráficos en {plots}")


if __name__ == "__main__":
    main()
