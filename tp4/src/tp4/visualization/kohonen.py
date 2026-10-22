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
