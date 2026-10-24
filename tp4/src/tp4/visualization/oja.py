from typing import Sequence

from tp4.types.float import Float
from tp4.visualization.pca import _BLUE, _INK, _MUTED, _ORANGE, _clean

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def plot_convergence(cosines: npt.NDArray[Float], norms: npt.NDArray[Float]) -> Figure:
    """|cos(w, PC1)| and ‖w‖ at the end of every epoch."""
    epochs = np.arange(1, len(cosines) + 1)
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    top.plot(epochs, cosines, color=_BLUE, linewidth=2)
    top.axhline(1, color=_MUTED, linestyle="--", linewidth=1)
    top.set_ylabel("|cos(w, PC1)|")
    top.set_title(f"Convergence of w — final |cos| = {cosines[-1]:.5f}, ‖w‖ = {norms[-1]:.4f}")
    _clean(top)

    bottom.plot(epochs, norms, color=_ORANGE, linewidth=2)
    bottom.axhline(1, color=_MUTED, linestyle="--", linewidth=1)
    bottom.set_xlabel("Epoch")
    bottom.set_ylabel("‖w‖")
    _clean(bottom)

    figure.tight_layout()
    return figure

def plot_loadings(weights: npt.NDArray[Float], component: npt.NDArray[Float], variables: Sequence[str]) -> Figure:
    """Weights of the Oja neuron next to the PC1 loadings, one pair of bars per variable sorted by |PC1|."""
    order = np.argsort(np.abs(component))
    rows = np.arange(len(variables))
    figure, ax = plt.subplots(figsize=(9, 0.32 * len(variables) + 1.5))

    ax.barh(rows + 0.2, component[order], height=0.4, color=_BLUE, label="PC1 (eigenvector)")
    ax.barh(rows - 0.2, weights[order], height=0.4, color=_ORANGE, label="Oja w")
    ax.set_yticks(rows, np.asarray(variables)[order], fontsize=8)
    ax.axvline(0, color=_MUTED, linewidth=0.8)
    ax.set_xlabel("Weight / loading")
    ax.set_title("Oja weights vs PC1 loadings")
    ax.legend(frameon=False, loc="lower right")
    _clean(ax)

    figure.tight_layout()
    return figure

def plot_index_comparison(index: npt.NDArray[Float], scores: npt.NDArray[Float], labels: Sequence[str] | None) -> Figure:
    """Oja index vs PC1 score of every sample; on the identity line when both agree."""
    figure, ax = plt.subplots(figsize=(9, 8))

    limit = 1.1 * max(np.abs(index).max(), np.abs(scores).max())
    ax.plot([-limit, limit], [-limit, limit], color=_MUTED, linestyle="--", linewidth=1, label="Identity")
    ax.scatter(scores, index, s=36, color=_BLUE, edgecolor="white", linewidth=1, zorder=3)
    if labels is not None:
        for x, y, label in zip(scores, index, labels):
            ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7, color=_INK)

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("PC1 score")
    ax.set_ylabel("Oja index")
    ax.set_title(f"Oja index vs PC1 (r = {np.corrcoef(index, scores)[0, 1]:.5f})")
    ax.legend(frameon=False, loc="upper left")
    _clean(ax)

    figure.tight_layout()
    return figure
