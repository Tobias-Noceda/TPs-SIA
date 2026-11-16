from typing import List, Sequence, Tuple

from tp5.types.float import Float
from tp5.visualization.glyphs import draw_glyph

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Figures of the ex. 1b study. The evaluations come as a DataFrame with one row per
# (variant, seed, test_level): input_pixels, output_pixels (mean wrong pixels) and recovered
# (fraction of outputs within the tolerance).

def _lines(ax: Axes, evaluations: pd.DataFrame, column: str, scale: float = 1.0) -> None:
    """One line per variant: mean over the seeds, band = min–max over the seeds."""
    for variant in dict.fromkeys(evaluations["variant"]):
        seeds = evaluations[evaluations["variant"] == variant].groupby("test_level")[column].agg(["mean", "min", "max"]) * scale
        line, = ax.plot(seeds.index, seeds["mean"], marker="o", markersize=3, label=variant)
        ax.fill_between(seeds.index, seeds["min"], seeds["max"], color=line.get_color(), alpha=0.15)

def _input_line(ax: Axes, evaluations: pd.DataFrame) -> None:
    """Wrong pixels of the noisy input: below this line the model removes noise, above it adds errors."""
    damage = evaluations.groupby("test_level")["input_pixels"].mean()
    ax.plot(damage.index, damage, color="black", linestyle="--", linewidth=1.2, label="noisy input")

def plot_denoising_curves(study: str, evaluations: pd.DataFrame, noise: str, tolerance: int) -> Figure:
    """Wrong pixels left in the output and outputs within the tolerance, against the noise of the input."""
    fig, (ax_pixels, ax_recovered) = plt.subplots(1, 2, figsize=(12, 4.5))
    _lines(ax_pixels, evaluations, "output_pixels")
    _input_line(ax_pixels, evaluations)
    ax_pixels.set_ylabel("Mean wrong pixels (of 35)")
    ax_pixels.set_title("Wrong pixels: output vs noisy input")
    ax_pixels.legend(title=study, fontsize=8)

    _lines(ax_recovered, evaluations, "recovered", 100)
    ax_recovered.set_ylim(0, 102)
    ax_recovered.set_ylabel(f"Outputs with ≤ {tolerance} wrong px (%)")
    ax_recovered.set_title("Recovered patterns")

    for ax in (ax_pixels, ax_recovered):
        ax.set_xlabel(f"Noise level of the input ({noise})")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{study}: mean and min–max over seeds")
    fig.tight_layout()
    return fig

def plot_level_heatmap(evaluations: pd.DataFrame, noise: str) -> Figure:
    """Mean wrong pixels of the output for every (training level, test level)."""
    table = evaluations.pivot_table(index="variant", columns="test_level", values="output_pixels", aggfunc="mean", sort=False)
    damage = evaluations.groupby("test_level")["input_pixels"].mean()

    fig, ax = plt.subplots(figsize=(1.1 * len(table.columns) + 2.5, 0.6 * len(table) + 2.2))
    image = ax.imshow(table.to_numpy(), cmap="viridis_r", aspect="auto")
    for row in range(table.shape[0]):
        for column in range(table.shape[1]):
            value = table.iat[row, column]
            ax.text(column, row, f"{value:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if value > table.to_numpy().max() / 2 else "black")
    ax.set_xticks(range(len(table.columns)), [f"{level:g}\n({damage[level]:.1f} px)" for level in table.columns])
    ax.set_yticks(range(len(table)), list(table.index))
    ax.set_xlabel(f"Noise level of the input, {noise} (mean wrong pixels of the input)")
    ax.set_ylabel("Trained with")
    ax.set_title("Mean wrong pixels of the output")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    return fig

def plot_noise_types(evaluations: pd.DataFrame) -> Figure:
    """One panel per noise of the input: how the models trained with each noise type clean it."""
    kinds = list(dict.fromkeys(evaluations["test_type"]))
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.2 * len(kinds), 4.5), squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        subset = evaluations[evaluations["test_type"] == kind]
        _lines(ax, subset, "output_pixels")
        _input_line(ax, subset)
        ax.set_xlabel(f"Noise level of the input ({kind})")
        ax.set_title(f"Input corrupted with {kind}")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Mean wrong pixels (of 35)")
    axes[0][0].legend(title="trained with", fontsize=8)
    fig.suptitle("noise_type: mean and min–max over seeds")
    fig.tight_layout()
    return fig

def plot_examples(
    originals: npt.NDArray[Float],
    labels: Sequence[str],
    shape: Tuple[int, int],
    rows: List[Tuple[float, npt.NDArray[Float], npt.NDArray[Float]]],
    title: str,
) -> Figure:
    """
    Original patterns (top) and, per noise level, the noisy input and the binarized output below it.
    Pixels that differ from the original are outlined in red.

    Parameters:
    rows : (level, noisy inputs, outputs) per noise level, each (patterns, pixels).
    """
    fig, axes = plt.subplots(1 + 2 * len(rows), len(labels), figsize=(1.15 * len(labels) + 1.2, 1.55 * (1 + 2 * len(rows))), squeeze=False)
    for column, (original, label) in enumerate(zip(originals, labels)):
        draw_glyph(axes[0, column], original, shape, title=label)
    axes[0, 0].set_ylabel("original", fontsize=8)

    for index, (level, noisy, outputs) in enumerate(rows):
        top, bottom = axes[1 + 2 * index], axes[2 + 2 * index]
        for column, original in enumerate(originals):
            for ax, values in ((top[column], noisy[column]), (bottom[column], outputs[column])):
                wrong = (values >= 0.5) != (original >= 0.5)
                # The input is drawn as it is (gaussian noise is continuous), the output binarized
                draw_glyph(ax, values if ax is top[column] else values >= 0.5, shape, wrong,
                           title=f"{int(wrong.sum())} px", title_color="red" if wrong.any() else "green")
        top[0].set_ylabel(f"noisy {level:g}", fontsize=8)
        bottom[0].set_ylabel("output", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig

def plot_confusion(confusion: npt.NDArray[Float], labels: Sequence[str], title: str) -> Figure:
    """Row = clean pattern that was corrupted, column = training pattern closest to the output (fractions per row)."""
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    image = ax.imshow(confusion, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), list(labels), fontsize=8)
    ax.set_yticks(range(len(labels)), list(labels), fontsize=8)
    ax.set_xlabel("Training pattern closest to the output")
    ax.set_ylabel("Corrupted pattern")
    ax.set_title(title, fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig
