from typing import Dict, Sequence

from tp4.types.float import Float
from tp4.visualization.kohonen import CATEGORIES
from tp4.visualization.pca import _INK, _MUTED, _clean

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Plots of exercise 1 that color every creator by its dominant topic (CATEGORIES order = names order)

def _compact(value: float) -> str:
    """283000 → 283k, 1.6e6 → 1.6M, small values with 3 significant digits."""
    for limit, suffix in ((1e6, "M"), (1e3, "k")):
        if abs(value) >= limit:
            return f"{value / limit:.3g}{suffix}"
    return f"{value:.3g}"

def _legend(ax, names: Sequence[str], **kwargs) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORIES[i]) for i in range(len(names))]
    ax.legend(handles, names, frameon=False, fontsize=8, **kwargs)

def plot_topic_shares(shares: npt.NDArray[Float], categories: npt.NDArray[np.intp], names: Sequence[str], labels: Sequence[str]) -> Figure:
    """Stacked share of each topic per creator, grouped by dominant topic and sorted by its share."""
    order = np.lexsort((-shares[np.arange(len(shares)), categories], categories))
    figure, ax = plt.subplots(figsize=(max(10, 0.25 * len(order)), 4.8))

    bottom = np.zeros(len(order))
    for c, name in enumerate(names):
        ax.bar(range(len(order)), shares[order, c], bottom=bottom, color=CATEGORIES[c], width=0.85, label=name)
        bottom += shares[order, c]

    counts = np.bincount(categories, minlength=len(names))
    ax.set_xticks(range(len(order)), np.asarray(labels)[order], rotation=90, fontsize=7)
    ax.set_ylabel("Share of posts (%)")
    ax.set_title("Topics per creator, grouped by dominant topic — " + ", ".join(f"{n}: {c}" for n, c in zip(names, counts)))
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, ncol=len(names), loc="upper center", bbox_to_anchor=(0.5, -0.32))
    ax.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    return figure

def plot_scores_by_topic(scores: npt.NDArray[Float], ratio: npt.NDArray[Float], categories: npt.NDArray[np.intp], names: Sequence[str], labels: Sequence[str], sizes: npt.NDArray[Float] | None = None) -> Figure:
    """PC1 vs PC2 scores colored by dominant topic; marker area ∝ sizes (e.g. followers) when given."""
    figure, ax = plt.subplots(figsize=(11, 8.5))

    area = 36 if sizes is None else 20 + 380 * sizes / sizes.max()
    ax.scatter(scores[:, 0], scores[:, 1], s=area, c=[CATEGORIES[c] for c in categories], edgecolor="white", linewidth=0.8, alpha=0.9, zorder=3)
    for (x, y), label in zip(scores[:, :2], labels):
        ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=6.5, color=_INK)

    ax.axhline(0, color=_MUTED, linewidth=0.8)
    ax.axvline(0, color=_MUTED, linewidth=0.8)
    ax.set_xlabel(f"PC1 ({ratio[0]:.1%})")
    ax.set_ylabel(f"PC2 ({ratio[1]:.1%})")
    ax.set_title("PC1 / PC2 scores by dominant topic" + (" (area ∝ followers)" if sizes is not None else ""))
    _legend(ax, names, loc="best")
    _clean(ax)

    figure.tight_layout()
    return figure

def plot_index_by_topic(index: npt.NDArray[Float], categories: npt.NDArray[np.intp], names: Sequence[str], labels: Sequence[str], name: str) -> Figure:
    """A 1-D index per creator, sorted, each bar colored by the creator's dominant topic."""
    order = np.argsort(index)[::-1]
    figure, ax = plt.subplots(figsize=(max(10, 0.25 * len(index)), 5))

    ax.bar(range(len(index)), index[order], color=[CATEGORIES[c] for c in categories[order]])
    ax.set_xticks(range(len(index)), np.asarray(labels)[order], rotation=90, fontsize=7)
    ax.axhline(0, color=_MUTED, linewidth=0.8)
    ax.set_ylabel(name)
    ax.set_title(f"{name} per creator, colored by dominant topic")
    _legend(ax, names, loc="upper right")
    _clean(ax)

    figure.tight_layout()
    return figure

def plot_index_vs(index: npt.NDArray[Float], values: npt.NDArray[Float], categories: npt.NDArray[np.intp], names: Sequence[str], labels: Sequence[str], index_name: str, values_name: str, log: bool = False) -> Figure:
    """An index against another variable (e.g. followers), colored by dominant topic."""
    figure, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(values, index, s=36, c=[CATEGORIES[c] for c in categories], edgecolor="white", linewidth=0.8, zorder=3)
    for x, y, label in zip(values, index, labels):
        ax.annotate(label, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=6.5, color=_INK)
    if log:
        ax.set_xscale("log")

    ranks = lambda v: np.argsort(np.argsort(v))
    rho = np.corrcoef(ranks(index), ranks(values))[0, 1]
    ax.axhline(0, color=_MUTED, linewidth=0.8)
    ax.set_xlabel(values_name + (" (log)" if log else ""))
    ax.set_ylabel(index_name)
    ax.set_title(f"{index_name} vs {values_name} (Spearman ρ = {rho:+.2f})")
    _legend(ax, names, loc="best")
    _clean(ax)

    figure.tight_layout()
    return figure

def plot_strategy_comparison(comparison: pd.DataFrame, metrics: Dict[str, str], title: str) -> Figure:
    """One small bar chart per metric, one bar per strategy (rows of comparison)."""
    columns = min(4, len(metrics))
    rows = int(np.ceil(len(metrics) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.6 * columns, 3.1 * rows), squeeze=False)
    strategies = list(comparison.index)
    colors = [CATEGORIES[i % len(CATEGORIES)] for i in range(len(strategies))]

    for ax, (metric, label) in zip(axes.flat, metrics.items()):
        values = comparison[metric].astype(float).to_numpy()
        ax.bar(range(len(strategies)), values, color=colors)
        for x, value in enumerate(values):
            ax.text(x, value, _compact(value), ha="center", va="bottom", fontsize=7, color=_INK)
        ax.set_xticks(range(len(strategies)), strategies, rotation=30, ha="right", fontsize=7)
        ax.set_title(label, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
    for ax in list(axes.flat)[len(metrics):]:
        ax.set_axis_off()

    figure.suptitle(title)
    figure.tight_layout()
    return figure
