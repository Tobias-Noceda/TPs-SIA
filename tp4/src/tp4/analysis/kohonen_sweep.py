import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Summary and plots of experiments/kohonen_sweep.py
parser = argparse.ArgumentParser(description="Kohonen hyperparameter sweep analysis.")

parser.add_argument("-i", "--input", dest="input", type=str, default="res/output/kohonen_sweep", help="Sweep output directory")

# Categorical slots in fixed order, one per initial η
_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

def _load(runs: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(runs.glob("*.npz")):
        run = np.load(path)
        final = run["results"][-1]
        rows.append({
            "grid_size": int(run["grid_size"]),
            "learning_rate": float(run["learning_rate"]),
            "radius": str(run["radius"]),
            "seed": int(run["seed"]),
            "quantization_error": float(final[3]),
            "topographic_error": float(final[4]),
            "dead_neurons": int((run["hits"] == 0).sum()),
            "dead_share": float((run["hits"] == 0).mean()),
            "mean_u_matrix": float(run["u_matrix"].mean()),
        })
    return pd.DataFrame(rows)

def analyze(directory: Path) -> pd.DataFrame:
    """Write summary.csv (mean and std over seeds per combination) and plots/sweep.png; return the summary."""
    runs = _load(directory / "runs")
    if runs.empty:
        raise FileNotFoundError(f"No runs found in {directory / 'runs'}")

    metrics = ["quantization_error", "topographic_error", "dead_neurons", "dead_share", "mean_u_matrix"]
    summary = runs.groupby(["grid_size", "learning_rate", "radius"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values(["quantization_error_mean", "topographic_error_mean"])
    summary.to_csv(directory / "summary.csv", index=False)
    print(summary.head(10).to_string(index=False))

    plots = directory / "plots"
    plots.mkdir(exist_ok=True)

    # One row per initial R; final QE, TE and dead share vs grid size, one line per initial η (mean ± std over seeds)
    radii = sorted(summary["radius"].unique())
    etas = sorted(summary["learning_rate"].unique())
    columns = [("quantization_error", "Quantization error"), ("topographic_error", "Topographic error"), ("dead_share", "Dead neurons (share)")]
    figure, axes = plt.subplots(len(radii), len(columns), figsize=(5 * len(columns), 3.4 * len(radii)), squeeze=False)

    for row, radius in zip(axes, radii):
        for ax, (metric, title) in zip(row, columns):
            for color, eta in zip(_SERIES, etas):
                data = summary[(summary["radius"] == radius) & (summary["learning_rate"] == eta)].sort_values("grid_size")
                ax.errorbar(data["grid_size"], data[f"{metric}_mean"], yerr=data[f"{metric}_std"],
                            color=color, linewidth=2, marker="o", markersize=5, capsize=3, label=f"η₀ = {eta:g}")
            ax.set_title(f"{title} — R₀ = {radius}")
            ax.set_xlabel("Grid size k")
            ax.set_xticks(sorted(summary["grid_size"].unique()))
            ax.grid(alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(frameon=False)

    figure.tight_layout()
    figure.savefig(plots / "sweep.png", dpi=150)
    print(f"Saved {directory / 'summary.csv'} and {plots / 'sweep.png'}")
    plt.close(figure)

    return summary

def main():
    args = parser.parse_args()
    analyze(Path(args.input))
