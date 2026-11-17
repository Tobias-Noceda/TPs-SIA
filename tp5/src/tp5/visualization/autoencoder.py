from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

from tp5.functions.pixel_error import pixel_errors
from tp5.types.epoch_result import EpochResult
from tp5.types.float import Float
from tp5.types.samples import Samples
from tp5.types.training_run import TrainingRun
from tp5.visualization.glyphs import draw_glyph

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

type Decoder = Callable[[Samples], Samples]

def plot_history(history: Sequence[EpochResult], patterns: int, goal_pixels: int | None, learned_pixels: int | None = None) -> Figure:
    """
    Loss, max wrong pixels and learned patterns per epoch.

    Parameters:
    goal_pixels    : goal line of the pixel panel (None = no line).
    learned_pixels : wrong pixels allowed when the history counted the learned patterns, i.e. the
                     goal of the training config (None = 0). Defaults to goal_pixels.
    """
    if learned_pixels is None:
        learned_pixels = goal_pixels
    epochs, loss, max_errors, learned = np.array(history, dtype=np.float64).reshape(-1, 4).T

    # Stacked with a shared epoch axis: the pixel error can be read against the loss at the same epoch
    fig, (ax_loss, ax_pixels) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax_loss.plot(epochs + 1, loss)
    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")

    ax_pixels.plot(epochs + 1, max_errors, label="Max wrong pixels")
    if goal_pixels is not None:
        ax_pixels.axhline(goal_pixels, color="gray", linestyle="--", linewidth=1, label=f"Goal ({goal_pixels} px)")
    ax_pixels.set_xlabel("Epoch")
    ax_pixels.set_ylabel("Wrong pixels")
    ax_learned = ax_pixels.twinx()
    ax_learned.plot(epochs + 1, learned, color="tab:green", label=f"Patterns ≤ {learned_pixels or 0} px")
    ax_learned.set_ylim(0, patterns + 1)
    ax_learned.set_ylabel("Learned patterns")
    handles = ax_pixels.get_legend_handles_labels()[0] + ax_learned.get_legend_handles_labels()[0]
    ax_pixels.legend(handles, [h.get_label() for h in handles], loc="center right")
    ax_pixels.set_title("Pixel error")

    fig.tight_layout()
    return fig

def plot_reconstructions(inputs: Samples, reconstructions: Samples, labels: List[str], shape: Tuple[int, int], goal_pixels: int | None) -> Figure:
    """Each original next to its reconstruction (binarized at 0.5), wrong pixels outlined in red."""
    errors = pixel_errors(reconstructions, inputs)
    binarized = (reconstructions >= 0.5).astype(np.float64)

    columns = 8
    rows = int(np.ceil(len(inputs) / columns))
    fig, axes = plt.subplots(2 * rows, columns, figsize=(columns * 1.1, rows * 2.6))
    for ax in axes.flat:
        ax.axis("off")
    for i, (original, output) in enumerate(zip(inputs, binarized)):
        row, column = divmod(i, columns)
        top, bottom = axes[2 * row, column], axes[2 * row + 1, column]
        top.axis("on")
        bottom.axis("on")
        draw_glyph(top, original, shape, title=labels[i])
        if goal_pixels is not None:
            color = "black" if errors[i] <= goal_pixels else "red"
        else:
            color = "black"
        draw_glyph(bottom, output, shape, wrong=output != original, title=f"{errors[i]} px", title_color=color)

    fig.suptitle("Originals (top) and reconstructions (bottom)")
    fig.tight_layout()
    return fig

def plot_latent(latent: Samples, labels: List[str]) -> Figure:
    """Latent code of every pattern, labeled with its character."""
    if latent.shape[1] != 2:
        raise ValueError(f"The latent space must be 2-D to plot it, got {latent.shape[1]} dimensions.")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(latent[:, 0], latent[:, 1], s=20)
    for (z1, z2), label in zip(latent, labels):
        ax.annotate(label, (z1, z2), textcoords="offset points", xytext=(4, 4), fontsize=11)
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_title("Patterns in the latent space")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

def nearest_patterns(generated: Samples, inputs: Samples) -> Tuple[npt.NDArray[np.intp], npt.NDArray[np.int64]]:
    """Index of the training pattern closest (Hamming, binarized at 0.5) to each generated one, and that distance."""
    generated_bits = generated[:, None, :] >= 0.5
    input_bits = inputs[None, :, :] >= 0.5
    distances = np.sum(generated_bits != input_bits, axis=2)
    nearest = np.argmin(distances, axis=1)
    return nearest, distances[np.arange(len(generated)), nearest]

def latent_grid(latent: Samples, size: int, padding: float = 0.1) -> Tuple[npt.NDArray[Float], npt.NDArray[Float]]:
    """Regular size × size grid over the bounding box of the latent codes, enlarged by `padding`."""
    low, high = latent.min(axis=0), latent.max(axis=0)
    margin = (high - low) * padding
    return np.linspace(low[0] - margin[0], high[0] + margin[0], size), np.linspace(low[1] - margin[1], high[1] + margin[1], size)

def plot_latent_grid(decode: Decoder, latent: Samples, inputs: Samples, labels: List[str], shape: Tuple[int, int], goal_pixels: int | None, size: int = 15) -> Figure:
    """
    Decode a regular grid of latent points. Each glyph is titled with its nearest training pattern
    and the Hamming distance to it; glyphs farther than goal_pixels from every pattern are new (blue).
    """
    z1, z2 = latent_grid(latent, size)
    # Top row = highest z2, so the mosaic is oriented like the latent space scatter
    points = np.array([(a, b) for b in z2[::-1] for a in z1])
    generated = decode(points)
    nearest, distances = nearest_patterns(generated, inputs)

    fig, axes = plt.subplots(size, size, figsize=(size * 0.8, size * 1.05))
    for ax, glyph, index, distance in zip(axes.flat, generated, nearest, distances):
        new = distance > (goal_pixels or 0)
        draw_glyph(ax, glyph, shape, title=f"{labels[index]}+{distance}" if new else labels[index], title_color="tab:blue" if new else "black")
    fig.suptitle(f"Decoded latent grid: z1 ∈ [{z1[0]:.2f}, {z1[-1]:.2f}], z2 ∈ [{z2[0]:.2f}, {z2[-1]:.2f}] (blue = new, nearest+distance)")
    fig.tight_layout()
    return fig

def default_pair(latent: Samples, inputs: Samples) -> Tuple[int, int]:
    """
    Two patterns that are neighbours in the latent space but very different in pixels: the points
    between them are the most likely to decode into something new.
    """
    latent_distances = np.linalg.norm(latent[:, None, :] - latent[None, :, :], axis=2)
    np.fill_diagonal(latent_distances, np.inf)
    neighbours = np.argmin(latent_distances, axis=1)
    pixel_distances = np.sum((inputs >= 0.5) != (inputs[neighbours] >= 0.5), axis=1)
    first = int(np.argmax(pixel_distances))
    return first, int(neighbours[first])

def plot_interpolation(decode: Decoder, latent: Samples, inputs: Samples, labels: List[str], shape: Tuple[int, int], goal_pixels: int | None, pair: Tuple[int, int], steps: int = 9) -> Figure:
    """Decode the segment between the latent codes of two patterns."""
    a, b = pair
    t = np.linspace(0, 1, steps)
    points = (1 - t)[:, None] * latent[a] + t[:, None] * latent[b]
    generated = decode(points)
    nearest, distances = nearest_patterns(generated, inputs)

    fig, axes = plt.subplots(1, steps, figsize=(steps * 1.2, 2.4))
    for ax, glyph, step, index, distance in zip(axes, generated, t, nearest, distances):
        new = distance > (goal_pixels or 0)
        name = f"{labels[index]}+{distance}" if new else labels[index]
        draw_glyph(ax, glyph, shape, title=f"t={step:.2f}\n{name}", title_color="tab:blue" if new else "black")
    fig.suptitle(f"Interpolation {labels[a]} → {labels[b]} in the latent space (blue = new, nearest+distance)")
    fig.tight_layout()
    return fig

def run_figures(run: TrainingRun, pair: Tuple[int, int] | None = None) -> Dict[str, Figure]:
    """Every figure of a basic autoencoder run, by name."""
    latent = run.model.encode(run.inputs)
    reconstructions = run.model.reconstruct(run.inputs)
    pair = pair if pair is not None else default_pair(latent, run.inputs)
    shape = (run.shape.height, run.shape.width)

    figures = {
        # The learned patterns of the history were counted with the goal of the training config, which
        # can differ from run.goal_pixels (ej1a-analysis draws its own goal over runs trained without one)
        "history": plot_history(run.history, len(run.inputs), run.goal_pixels, run.source.get("goal_pixels", 0)),
        "reconstructions": plot_reconstructions(run.inputs, reconstructions, run.labels, shape, run.goal_pixels),
    }
    if latent.shape[1] == 2:
        figures["latent_space"] = plot_latent(latent, run.labels)
        figures["latent_grid"] = plot_latent_grid(run.model.decode, latent, run.inputs, run.labels, shape, run.goal_pixels)
    figures["interpolation"] = plot_interpolation(run.model.decode, latent, run.inputs, run.labels, shape, run.goal_pixels, pair)
    return figures

def plot_run(run: TrainingRun, save_dir: str | None = None) -> None:
    """Show every figure of the run, or save them as PNG into `save_dir`."""
    figures = run_figures(run)
    if save_dir is None:
        plt.show()
        return

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        figure.savefig(Path(save_dir) / f"{name}.png", dpi=130)
        plt.close(figure)
