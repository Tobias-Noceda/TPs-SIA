from typing import Sequence

from tp4.types.float import Float
from tp4.types.hopfield_recall import HopfieldRecall
from tp4.types.pattern import Patterns
from tp4.visualization.pca import _BLUE, _ORANGE, _clean

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Recalls with more steps than this only show the first ones and the final state
_MAX_COLUMNS = 8

def _draw(ax, pattern: npt.NDArray[Float], shape: tuple[int, int], title: str) -> None:
    # 1 (active) black, -1 (inactive) white, like the source images
    ax.imshow(pattern.reshape(shape), cmap="gray_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

def plot_patterns(patterns: Patterns, shape: tuple[int, int], names: Sequence[str]) -> Figure:
    """Stored patterns side by side."""
    figure, axes = plt.subplots(1, len(names), figsize=(2 * len(names), 2.4), squeeze=False)
    for ax, pattern, name in zip(axes[0], patterns, names):
        _draw(ax, pattern, shape, name)

    figure.suptitle("Stored patterns")
    figure.tight_layout()
    return figure

def plot_recalls(recalls: Sequence[HopfieldRecall], levels: Sequence[Float], shape: tuple[int, int], results: Sequence[str]) -> Figure:
    """One row per noise level: S(0) = query, S(1), ... up to S(t_final)."""
    columns = min(max(len(recall.steps) for recall in recalls), _MAX_COLUMNS)
    figure, axes = plt.subplots(len(recalls), columns, figsize=(1.6 * columns + 1, 1.8 * len(recalls)), squeeze=False)

    for row, (recall, level, result) in zip(axes, zip(recalls, levels, results)):
        steps = list(recall.steps)
        if len(steps) > columns:
            steps = steps[:columns - 1] + steps[-1:]
        for ax, step in zip(row, steps):
            _draw(ax, step.state, shape, f"t = {step.step}\nH = {step.energy:.1f}")
        for ax in row[len(steps):]:
            ax.axis("off")
        row[0].set_ylabel(f"noise {level:.0%}\n{recall.convergence.value}\n{result}", fontsize=8)

    figure.suptitle("State per step for each noise level")
    figure.tight_layout()
    return figure

def plot_energy(recalls: Sequence[HopfieldRecall], levels: Sequence[Float]) -> Figure:
    """Energy H(S(t)) per step, one line per noise level."""
    figure, ax = plt.subplots(figsize=(9, 4.5))
    colors = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(recalls)))

    for recall, level, color in zip(recalls, levels, colors):
        ax.plot([s.step for s in recall.steps], [s.energy for s in recall.steps], marker="o", color=color, linewidth=2, label=f"noise {level:.0%}")

    ax.set_xlabel("Step t")
    ax.set_ylabel("Energy H")
    ax.set_title("Energy per step")
    ax.legend(frameon=False, fontsize=8)
    _clean(ax)

    figure.tight_layout()
    return figure

def plot_overlap_by_noise(levels: Sequence[Float], overlaps: npt.NDArray[Float], names: Sequence[str]) -> Figure:
    """Overlap m = ξ · S(t_final) / N of the final state with each stored pattern, per noise level."""
    figure, ax = plt.subplots(figsize=(9, 4.5))
    colors = [_BLUE, _ORANGE] + list(plt.get_cmap("tab10").colors[2:])

    for k, name in enumerate(names):
        ax.plot(levels, overlaps[:, k], marker="o", color=colors[k % len(colors)], linewidth=2, label=name)

    ax.axhline(0, color="#6b6b66", linewidth=0.8)
    ax.set_xlabel("Noise (fraction of flipped entries)")
    ax.set_ylabel("Overlap with the final state")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Overlap of the recalled state with each stored pattern")
    ax.legend(frameon=False, fontsize=8)
    _clean(ax)

    figure.tight_layout()
    return figure
