from typing import Sequence

from tp4.types.float import Float

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

# Diverging blue ↔ red with a gray midpoint at 0 (negative ↔ positive correlation / loading)
DIVERGING = LinearSegmentedColormap.from_list("diverging", ["#1c5cab", "#f0efec", "#c0392f"])

_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_INK = "#1f2328"
_MUTED = "#6b6b66"

def _clean(ax) -> None:
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

def plot_heatmap(matrix: npt.NDArray[Float], rows: Sequence[str], columns: Sequence[str], title: str, annotate: bool = True) -> Figure:
    """Diverging heatmap symmetric around 0, e.g. a correlation matrix or the loadings."""
    limit = max(float(np.abs(matrix).max()), 1e-12)
    figure, ax = plt.subplots(figsize=(0.45 * len(columns) + 4, 0.35 * len(rows) + 2))

    image = ax.imshow(matrix, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(columns)), columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    ax.set_title(title)
    figure.colorbar(image, ax=ax, fraction=0.03, pad=0.02)

    if annotate:
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(value) > 0.6 * limit else _INK)

    figure.tight_layout()
    return figure

def plot_scree(eigenvalues: npt.NDArray[Float], components: int) -> Figure:
    """Explained variance per component (bars) and cumulative (line); kept components highlighted."""
    ratio = eigenvalues / eigenvalues.sum()
    indices = np.arange(1, len(eigenvalues) + 1)
    figure, ax = plt.subplots(figsize=(10, 4.5))

    ax.bar(indices, ratio, color=[_BLUE if i <= components else "#b7d3f6" for i in indices], label="Per component")
    ax.plot(indices, np.cumsum(ratio), color=_ORANGE, linewidth=2, marker="o", markersize=5, label="Cumulative")
    # Kaiser: an eigenvalue of 1 explains 1/p of the total variance of p standardized variables
    ax.axhline(1 / len(eigenvalues), color=_MUTED, linestyle="--", linewidth=1, label="λ = 1 (Kaiser)")

    ax.set_xticks(indices)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (share)")
    ax.set_title(f"Explained variance — {components} components kept ({np.cumsum(ratio)[components - 1]:.0%})")
    ax.legend(frameon=False)
    _clean(ax)

    figure.tight_layout()
    return figure

def plot_biplot(scores: npt.NDArray[Float], loadings: npt.NDArray[Float], ratio: npt.NDArray[Float], variables: Sequence[str], labels: Sequence[str] | None) -> Figure:
    """PC1 vs PC2 scores of each sample plus one arrow per variable (its loadings, scaled to the scores)."""
    figure, ax = plt.subplots(figsize=(12, 9))

    ax.scatter(scores[:, 0], scores[:, 1], s=36, color=_BLUE, edgecolor="white", linewidth=1, zorder=3)
    if labels is not None:
        for (x, y), label in zip(scores[:, :2], labels):
            ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7, color=_INK)

    # One scale for both axes (keeps the arrows' angles) chosen so every arrow fits inside the scores' spread
    spread = np.abs(scores[:, :2]).max(axis=0)
    scale = 0.85 * np.min(spread / np.maximum(np.abs(loadings[:, :2]).max(axis=0), 1e-12))
    tips = loadings[:, :2] * scale
    for (x, y), variable in zip(tips, variables):
        ax.annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=_ORANGE, linewidth=1.2))
        ax.text(x * 1.08, y * 1.08, variable, fontsize=7, color="#a8441c", ha="center", va="center")

    # Annotations don't resize the axes: fit scores and arrow labels explicitly
    limits = 1.15 * np.maximum(spread, np.abs(tips).max(axis=0) * 1.08)
    ax.set_xlim(-limits[0], limits[0])
    ax.set_ylim(-limits[1], limits[1])

    ax.axhline(0, color=_MUTED, linewidth=0.8)
    ax.axvline(0, color=_MUTED, linewidth=0.8)
    ax.set_xlabel(f"PC1 ({ratio[0]:.1%})")
    ax.set_ylabel(f"PC2 ({ratio[1]:.1%})")
    ax.set_title("Biplot: PC1 and PC2 scores with the variables' loadings")
    _clean(ax)

    figure.tight_layout()
    return figure

def plot_index(scores: npt.NDArray[Float], labels: Sequence[str] | None, component: int = 0) -> Figure:
    """Score of every sample on one component, sorted: the component used as an index."""
    values = scores[:, component]
    order = np.argsort(values)[::-1]
    names = np.asarray(labels)[order] if labels is not None else order.astype(str)
    figure, ax = plt.subplots(figsize=(max(10, 0.25 * len(values)), 5))

    ax.bar(range(len(values)), values[order], color=[_BLUE if v >= 0 else "#c0392f" for v in values[order]])
    ax.set_xticks(range(len(values)), names, rotation=90, fontsize=7)
    ax.axhline(0, color=_MUTED, linewidth=0.8)
    ax.set_ylabel(f"PC{component + 1}")
    ax.set_title(f"PC{component + 1} per sample")
    _clean(ax)

    figure.tight_layout()
    return figure
