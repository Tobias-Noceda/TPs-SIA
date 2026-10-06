"""
Ej. 3 — análisis de las corridas guardadas por `tp3.experiments.ej2_digits` con `res/config/ej3/studies.json`.

Cada corrida (784-256-10, oculta logística) entrenó con el conjunto de su variante y se evaluó en digits_test.csv.
Genera en `<out>/plots/`:
- dataset.png           accuracy de test por conjunto de entrenamiento, sin técnicas y con Xavier + desplazamientos.
- dataset_recall.png    recall de test por dígito según el conjunto de entrenamiento (con técnicas).
- lr_optimizer.png      accuracy de test final según η, una curva por optimizador.
- optimizer_curves.png  pérdida de train y accuracy de test por época, cada optimizador con su mejor η.
- techniques.png        accuracy de train y test de cada combinación de técnicas.
- confusion_test.png    matriz de confusión en test de la mejor configuración (normalizada por fila).
y `<out>/summary.csv` con las accuracies por configuración.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tp3.analysis.ej2_digits import (
    CLASSES, FAILED_ACCURACY, MUTED, SERIES, SHORT, Run, best_report, load_runs, recall_per_class, summary_table,
)
from tp3.experiments.ej2_digits import data_path, train_files

parser = argparse.ArgumentParser(description="Ej3: análisis del estudio de dígitos con more_digits.csv.")
parser.add_argument("-o", "--out", type=Path, default=Path("res/output/ej3_digits"), help="Directorio de resultados")

# Objective of the exercise
TARGET = 0.98


def target_line(ax: plt.Axes, horizontal: bool = True) -> None:
    line = ax.axhline if horizontal else ax.axvline
    line(TARGET, color=MUTED, linestyle="--", linewidth=1, label=f"objetivo {TARGET:.0%}")


def plot_dataset(table: pd.DataFrame, plots: Path) -> None:
    """Grouped bars: training set × (no techniques, Xavier + shifts)."""
    study = table[table["study"] == "dataset"].reset_index(drop=True)
    sets = list(dict.fromkeys(label.split(", ")[0] for label in study["variant"]))
    techniques = list(dict.fromkeys(label.split(", ")[1] for label in study["variant"]))
    x = np.arange(len(sets))
    width = 0.8 / len(techniques)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (color, technique) in enumerate(zip(SERIES, techniques)):
        rows = study.set_index("variant").loc[[f"{s}, {technique}" for s in sets]]
        offset = (i - (len(techniques) - 1) / 2) * width
        bars = ax.bar(x + offset, rows["test_accuracy"], width, yerr=rows["test_std"], color=color, capsize=3, label=technique)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in rows["test_accuracy"]], padding=4, fontsize=8)
    target_line(ax)
    ax.set_xticks(x, sets)
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("accuracy de test (época final)")
    ax.set_title("Conjunto de entrenamiento, 784-256-10 logística + Adam η = 0.001\n(media ± desvío entre semillas)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots / "dataset.png", dpi=150)
    plt.close(fig)


def plot_dataset_recall(table: pd.DataFrame, runs: dict[str, list[Run]], counts: dict[str, np.ndarray], plots: Path) -> None:
    """Per-digit test recall of each training set (with techniques), annotated with its training samples per digit."""
    study = table[(table["study"] == "dataset") & ~table["variant"].str.contains("sin técnicas")]
    x = np.arange(CLASSES)
    width = 0.8 / len(study)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (color, (_, row)) in enumerate(zip(SERIES, study.iterrows())):
        group = runs[row["config_id"]]
        recalls = np.array([recall_per_class(r.labels, r.predictions) for r in group])
        name = row["variant"].split(", ")[0]
        offset = (i - (len(study) - 1) / 2) * width
        ax.bar(x + offset, recalls.mean(axis=0), width, color=color,
               label=f"{name} (5: {counts[row['config_id']][5]}, 8: {counts[row['config_id']][8]} muestras)")
    ax.set_xticks(x)
    ax.set_xlabel("dígito")
    ax.set_ylabel("recall de test")
    ax.set_ylim(0, 1.05)
    ax.set_title("Recall por dígito según el conjunto de entrenamiento (Xavier + desplazamientos, media entre semillas)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=len(study), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(plots / "dataset_recall.png", dpi=150)
    plt.close(fig)


def plot_lr_optimizer(table: pd.DataFrame, plots: Path) -> dict[str, str]:
    """Final test accuracy vs. η per optimizer. Returns the config of the best η per optimizer."""
    study = table[table["study"] == "optimizer"]
    learned = study["test_accuracy"] >= FAILED_ACCURACY
    bottom = study.loc[learned, "test_accuracy"].min() - 0.01
    best: dict[str, str] = {}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for color, (optimizer, rows) in zip(SERIES, study.groupby("optimizer", sort=False)):
        rows = rows.sort_values("learning_rate")
        shown = rows["test_accuracy"].clip(lower=bottom)
        ok = rows["test_accuracy"] >= FAILED_ACCURACY
        ax.plot(rows["learning_rate"], shown, color=color, marker="o", linewidth=2, label=SHORT[optimizer])
        ax.errorbar(rows["learning_rate"][ok], shown[ok], yerr=rows["test_std"][ok], color=color, linestyle="none", capsize=3)
        for _, row in rows[~ok].iterrows():
            ax.annotate(f"falla: {row['test_accuracy']:.2f}", (row["learning_rate"], bottom), textcoords="offset points",
                        xytext=(-6, 4), ha="right", fontsize=8, color=color)
        best[optimizer] = rows.loc[rows["test_accuracy"].idxmax(), "config_id"]
    target_line(ax)
    ax.set_xscale("log")
    ax.set_xlabel("tasa de aprendizaje η")
    ax.set_ylabel("accuracy de test (época final)")
    ax.set_ylim(bottom - 0.003, 0.995)
    ax.set_title("η × optimizador, Xavier + desplazamientos ±1 (media ± desvío entre semillas)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots / "lr_optimizer.png", dpi=150)
    plt.close(fig)
    return best


def plot_optimizer_curves(runs: dict[str, list[Run]], best: dict[str, str], plots: Path) -> None:
    fig, (loss_ax, accuracy_ax) = plt.subplots(1, 2, figsize=(11, 4.2))
    for color, (optimizer, cid) in zip(SERIES, best.items()):
        group = runs[cid]
        label = f"{SHORT[optimizer]}, η = {group[0].config['learning_rate']:g}"
        for ax, key in ((loss_ax, "train_loss"), (accuracy_ax, "test_accuracy")):
            values = np.array([r.history[key] for r in group])
            epochs = np.arange(1, values.shape[1] + 1)
            ax.plot(epochs, np.nanmean(values, axis=0), color=color, linewidth=2, label=label)
            ax.fill_between(epochs, np.nanmin(values, axis=0), np.nanmax(values, axis=0), color=color, alpha=0.15, linewidth=0)
    loss_ax.set_yscale("log")
    loss_ax.set_ylabel("error cuadrático ½Σ (train, sin desplazar)")
    target_line(accuracy_ax)
    accuracy_ax.set_ylabel("accuracy de test")
    accuracy_ax.set_ylim(0.9, 1.0)
    for ax in (loss_ax, accuracy_ax):
        ax.set_xlabel("época")
        ax.grid(alpha=0.3)
    accuracy_ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Convergencia con el mejor η de cada optimizador (media; banda = mín.–máx. entre semillas)")
    fig.tight_layout()
    fig.savefig(plots / "optimizer_curves.png", dpi=150)
    plt.close(fig)


def plot_techniques(table: pd.DataFrame, plots: Path) -> None:
    study = table[table["study"] == "techniques"].reset_index(drop=True)
    y = np.arange(len(study))[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hlines(y, study["test_accuracy"], study["train_accuracy"], color=MUTED, linewidth=1.5, zorder=1)
    ax.scatter(study["train_accuracy"], y, color=SERIES[1], s=45, zorder=2, label="train")
    ax.errorbar(study["test_accuracy"], y, xerr=study["test_std"], fmt="o", color=SERIES[0], markersize=7, capsize=3,
                zorder=3, label="test")
    for yi, (_, row) in zip(y, study.iterrows()):
        ax.annotate(f"{row['test_accuracy']:.4f}", (row["test_accuracy"], yi), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    target_line(ax, horizontal=False)
    ax.set_yticks(y, study["variant"], fontsize=8)
    ax.set_xlim(min(0.95, study["test_accuracy"].min() - 0.005), 1.002)
    ax.set_xlabel("accuracy (época final)")
    ax.set_title("Técnicas, entrenando con digits.csv ∪ more_digits.csv\n(media ± desvío en test)")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots / "techniques.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parser.parse_args()
    plots = args.out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.out)
    with open(args.out / "studies.json") as f:
        saved = json.load(f)
    studies = saved["studies"]

    # Training samples per digit of each configuration's training set
    counts: dict[str, np.ndarray] = {}
    for variant in studies["dataset"]["variants"]:
        config = saved["configs"][variant["config_id"]]
        files = train_files(saved["spec"], config)
        counts[variant["config_id"]] = np.bincount(np.load(data_path(saved["spec"], config, args.out))["y"].argmax(1), minlength=CLASSES)
        print(f"{variant['label']:45s} ({' + '.join(Path(f).name for f in files)}): " + " ".join(map(str, counts[variant["config_id"]]))
              + f"  total {counts[variant['config_id']].sum()}")

    table = summary_table(runs, studies)
    table.to_csv(args.out / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    for name, study in table.groupby("study", sort=False):
        print(f"\n== {studies[name]['title']} ==")
        print(study[["variant", "seeds", "diverged", "train_accuracy", "test_accuracy", "test_std"]].to_string(index=False, float_format="%.4f"))

    plot_dataset(table, plots)
    plot_dataset_recall(table, runs, counts, plots)
    best = plot_lr_optimizer(table, plots)
    plot_optimizer_curves(runs, best, plots)
    plot_techniques(table, plots)

    best_report(runs, table, plots)
    top = table.loc[table["test_accuracy"].idxmax()]
    every = [(r.final_test, r.seed) for r in runs[top["config_id"]]]
    peak = max(max(r.history["test_accuracy"]) for r in runs[top["config_id"]])
    print(f"mejor semilla: {max(every)[0]:.4f} (semilla {max(every)[1]}); máximo en alguna época: {peak:.4f}")
    print(f"semillas >= {TARGET:.0%}: {sum(a >= TARGET for a, _ in every)}/{len(every)}")
    print(f"\nGráficos en {plots}")


if __name__ == "__main__":
    main()
