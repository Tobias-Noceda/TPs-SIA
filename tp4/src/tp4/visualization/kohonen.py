from typing import List, Sequence, Tuple

from tp4.types.epoch_result import EpochResult
from tp4.types.float import Float
from tp4.types.grid import GridPositions
from tp4.types.topology import Topology

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.patches import RegularPolygon

# Single-hue sequential colormap: every map plot shows a magnitude (hits, distance, variable mean)
_COLORMAP = "Blues"

def draw_grid(ax: Axes, positions: GridPositions, topology: Topology, values: npt.NDArray, title: str, annotate: bool = False, label: str = "") -> None:
    """Draw one cell per neuron colored by its value, row 0 on top. annotate writes each value inside its cell."""
    cells = [
        RegularPolygon((x, y), numVertices=topology.vertices, radius=topology.cell_radius, orientation=topology.orientation)
        for x, y in positions
    ]
    collection = PatchCollection(cells, cmap=_COLORMAP, edgecolor="white", linewidth=1.5)
    collection.set_array(np.asarray(values, dtype=np.float64))
    ax.add_collection(collection)

    if annotate:
        # Dark text on light cells, white on dark ones
        norm = collection.norm
        for (x, y), value in zip(positions, values):
            ax.text(x, y, f"{value:g}", ha="center", va="center", fontsize=8,
                    color="white" if norm(value) > 0.6 else "#1f2328")

    ax.set_xlim(positions[:, 0].min() - 1, positions[:, 0].max() + 1)
    ax.set_ylim(positions[:, 1].max() + 1, positions[:, 1].min() - 1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title)
    plt.colorbar(collection, ax=ax, fraction=0.046, pad=0.04, label=label)

def plot_map(positions: GridPositions, topology: Topology, hits: npt.NDArray, u_matrix: npt.NDArray[Float]) -> Figure:
    """Samples per neuron next to the U-matrix."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 5.5))
    draw_grid(left, positions, topology, hits, "Samples per neuron", annotate=True, label="samples")
    draw_grid(right, positions, topology, u_matrix, "U-matrix", label="mean distance to neighbors")
    figure.tight_layout()
    return figure

def plot_components(positions: GridPositions, topology: Topology, weights: npt.NDArray[Float], names: Sequence[str]) -> Figure:
    """One map per input variable, colored by that variable in each neuron's weights."""
    columns = min(4, len(names))
    rows = int(np.ceil(len(names) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.6 * rows), squeeze=False)

    for ax, name, values in zip(axes.flat, names, weights.T):
        draw_grid(ax, positions, topology, values, name)
    for ax in list(axes.flat)[len(names):]:
        ax.set_axis_off()

    figure.tight_layout()
    return figure

def plot_training(results: Tuple[EpochResult, ...]) -> Figure:
    """Quantization and topographic error per epoch, as two charts (different scales)."""
    epochs = [r.epoch for r in results]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    series: List[Tuple[str, List[Float]]] = [
        ("Quantization error", [r.quantization_error for r in results]),
        ("Topographic error", [r.topographic_error for r in results]),
    ]
    for ax, (title, values) in zip(axes, series):
        ax.plot(epochs, values, linewidth=2, color="#2563eb")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    return figure

# Categorical slots in fixed order (one per category, e.g. the dominant topic)
CATEGORIES = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#eda100", "#6250d6", "#008300", "#e34948"]
_EMPTY = "#e8e8e6"

def _names_by_neuron(winners: npt.NDArray[np.intp], labels: Sequence[str], neurons: int) -> List[List[str]]:
    names: List[List[str]] = [[] for _ in range(neurons)]
    for winner, label in zip(winners, labels):
        names[int(winner)].append(label.lstrip("@"))
    return names

def _write_names(ax: Axes, positions: GridPositions, names: List[List[str]], light: npt.NDArray[np.bool_]) -> None:
    """Names of each neuron's samples inside its cell; at most 4 lines, the rest as '+n'."""
    for (x, y), group, is_light in zip(positions, names, light):
        if not group:
            continue
        lines = group[:4] + ([f"+{len(group) - 4}"] if len(group) > 4 else [])
        ax.text(x, y, "\n".join(lines), ha="center", va="center", fontsize=5.5,
                color="#1f2328" if is_light else "white", linespacing=1.1)

def plot_labels(positions: GridPositions, topology: Topology, values: npt.NDArray[Float], winners: npt.NDArray[np.intp], labels: Sequence[str], title: str, label: str = "mean distance to neighbors") -> Figure:
    """Map colored by a value per neuron (e.g. the U-matrix) with the samples' names written in their winner."""
    k = int(round(np.sqrt(len(positions))))
    figure, ax = plt.subplots(figsize=(1.9 * k + 2, 1.6 * k + 1))
    draw_grid(ax, positions, topology, values, title, label=label)

    norm = plt.Normalize(float(np.min(values)), float(np.max(values)))
    _write_names(ax, positions, _names_by_neuron(winners, labels, len(positions)), norm(values) <= 0.6)

    figure.tight_layout()
    return figure

def plot_categories(positions: GridPositions, topology: Topology, categories: npt.NDArray[np.intp], names: Sequence[str], winners: npt.NDArray[np.intp], labels: Sequence[str], title: str) -> Figure:
    """
    Map with each neuron colored by a category (index into names, −1 = no category, e.g. a dead
    neuron) and the samples' names written in their winner.
    """
    k = int(round(np.sqrt(len(positions))))
    figure, ax = plt.subplots(figsize=(1.9 * k + 2, 1.6 * k + 1))

    colors = [CATEGORIES[c] if c >= 0 else _EMPTY for c in categories]
    cells = [
        RegularPolygon((x, y), numVertices=topology.vertices, radius=topology.cell_radius, orientation=topology.orientation,
                       facecolor=color, edgecolor="white", linewidth=1.5)
        for (x, y), color in zip(positions, colors)
    ]
    ax.add_collection(PatchCollection(cells, match_original=True))
    _write_names(ax, positions, _names_by_neuron(winners, labels, len(positions)), np.asarray(categories) < 0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORIES[i]) for i in range(len(names))]
    ax.legend(handles, names, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=len(names), fontsize=8)

    ax.set_xlim(positions[:, 0].min() - 1, positions[:, 0].max() + 1)
    ax.set_ylim(positions[:, 1].max() + 1, positions[:, 1].min() - 1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title)

    figure.tight_layout()
    return figure

def plot_projected_map(neurons: npt.NDArray[Float], samples: npt.NDArray[Float], grid_distances: npt.NDArray[Float], hits: npt.NDArray[np.intp], categories: npt.NDArray[np.intp], names: Sequence[str], ratio: npt.NDArray[Float]) -> Figure:
    """
    The map drawn on the PC1/PC2 plane: each neuron's weights projected (size = hits, hollow = dead),
    an edge between neurons adjacent on the grid, and every sample colored by its category.
    """
    figure, ax = plt.subplots(figsize=(11, 8.5))

    adjacent = np.argwhere(np.triu((grid_distances > 0) & (grid_distances <= 1 + 1e-9)))
    for i, j in adjacent:
        ax.plot(neurons[[i, j], 0], neurons[[i, j], 1], color="#9a9a94", linewidth=0.8, zorder=1)
    alive = hits > 0
    ax.scatter(neurons[alive, 0], neurons[alive, 1], s=25 + 25 * hits[alive], facecolor="#1f2328", alpha=0.25, edgecolor="none", zorder=2, label="Neuron (size = creators)")
    ax.scatter(neurons[~alive, 0], neurons[~alive, 1], s=25, facecolor="none", edgecolor="#1f2328", linewidth=0.8, zorder=2, label="Dead neuron")

    for c, name in enumerate(names):
        mask = categories == c
        ax.scatter(samples[mask, 0], samples[mask, 1], s=30, color=CATEGORIES[c], edgecolor="white", linewidth=0.8, zorder=3, label=name)

    ax.axhline(0, color="#6b6b66", linewidth=0.6)
    ax.axvline(0, color="#6b6b66", linewidth=0.6)
    ax.set_xlabel(f"PC1 ({ratio[0]:.1%})")
    ax.set_ylabel(f"PC2 ({ratio[1]:.1%})")
    ax.set_title("Kohonen map projected on PC1/PC2")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    return figure
