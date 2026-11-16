from typing import Dict, List, Sequence, Tuple

from tp5.types.float import Float

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Circle, Ellipse

# Figures of the ex. 2 study. Samples are flattened images (h · w · c, channels-last, values in [0, 1]).

type Images = npt.NDArray[Float]
type Shape = Tuple[int, int, int]

def as_image(sample: npt.NDArray[Float], shape: Shape) -> npt.NDArray[Float]:
    """Flattened sample → (h, w, 3) array imshow can draw."""
    height, width, channels = shape
    image = np.clip(np.asarray(sample, dtype=np.float64).reshape(height, width, channels), 0, 1)
    return np.repeat(image, 3, axis=2) if channels == 1 else image

def _show(ax: Axes, sample: npt.NDArray[Float], shape: Shape, title: str | None = None) -> None:
    ax.imshow(as_image(sample, shape), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title, fontsize=7)

def plot_mosaic(samples: Images, shape: Shape, title: str, columns: int = 16) -> Figure:
    """Every sample of the dataset."""
    rows = int(np.ceil(len(samples) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 0.7, rows * 0.7 + 0.5), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, sample in zip(axes.ravel(), samples):
        ax.imshow(as_image(sample, shape), interpolation="nearest")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig

def plot_loss_terms(study: str, curves: Dict[str, Tuple[npt.NDArray[Float], npt.NDArray[Float] | None]]) -> Figure:
    """
    Reconstruction error and KL divergence per epoch of each variant.

    Parameters:
    curves : variant -> (reconstruction per epoch, mean KL per epoch or None for a plain autoencoder).
    """
    fig, (ax_reconstruction, ax_kl) = plt.subplots(1, 2, figsize=(12, 4.5))
    for variant, (reconstruction, kl) in curves.items():
        epochs = np.arange(1, len(reconstruction) + 1)
        line, = ax_reconstruction.plot(epochs, reconstruction, label=variant)
        if kl is not None:
            ax_kl.plot(epochs, kl, color=line.get_color(), label=variant)
    ax_reconstruction.set_title("Reconstruction error (z = μ)")
    ax_kl.set_title("KL(q(z|x) || N(0, I)) per sample (regularizer)")
    ax_kl.set_yscale("log")
    for ax in (ax_reconstruction, ax_kl):
        ax.set_xscale("log")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3, which="both")
        ax.legend(title=study, fontsize=8)
    fig.suptitle(f"{study}: the two terms of J = reconstruction + λ KL")
    fig.tight_layout()
    return fig

def plot_rows(rows: Sequence[Tuple[str, Images]], shape: Shape, title: str, titles: Sequence[Sequence[str]] | None = None) -> Figure:
    """One row of images per entry (label on the left); `titles[row][column]` goes above each image."""
    columns = max(len(images) for _, images in rows)
    fig, axes = plt.subplots(len(rows), columns, figsize=(columns * 0.95 + 1.2, len(rows) * (1.25 if titles else 1.0) + 0.6), squeeze=False)
    for row, (label, images) in enumerate(rows):
        for column in range(columns):
            ax = axes[row, column]
            if column < len(images):
                _show(ax, images[column], shape, titles[row][column] if titles else None)
            else:
                ax.axis("off")
        axes[row, 0].set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig

def plot_latent_maps(
    maps: Dict[str, Tuple[npt.NDArray[Float], npt.NDArray[Float] | None]], samples: Images, shape: Shape, title: str
) -> Figure:
    """
    Where each model puts the training samples in a 2-D latent space, each drawn as its own image.
    Gray circles: 1, 2 and 3 σ of the prior N(0, I), where new samples are drawn from. Ellipses: 1 σ(x)
    of each q(z|x).

    Parameters:
    maps : variant -> (codes (samples, 2), σ (samples, 2) or None for a plain autoencoder).
    """
    fig, axes = plt.subplots(1, len(maps), figsize=(5.4 * len(maps), 5.6), squeeze=False)
    for ax, (variant, (codes, sigma)) in zip(axes[0], maps.items()):
        for radius in (1, 2, 3):
            ax.add_patch(Circle((0, 0), radius, fill=False, color="gray", linestyle="--", linewidth=0.8, zorder=1))
        if sigma is not None:
            for (x, y), (width, height) in zip(codes, sigma):
                ax.add_patch(Ellipse((x, y), 2 * width, 2 * height, color="tab:blue", alpha=0.12, zorder=2))
        for (x, y), sample in zip(codes, samples):
            ax.add_artist(AnnotationBbox(OffsetImage(as_image(sample, shape), zoom=0.7, interpolation="nearest"), (x, y), frameon=False, zorder=3))
        limit = max(3.3, float(np.abs(codes).max()) * 1.08)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.set_title(variant, fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig

def plot_grid(decoded: Images, shape: Shape, axis: npt.NDArray[Float], title: str) -> Figure:
    """Decoder output over a regular grid of a 2-D latent space (rows: z2 from high to low, columns: z1)."""
    size = len(axis)
    height, width, _ = shape
    mosaic = np.ones((size * height, size * width, 3))
    for index, sample in enumerate(decoded):
        row, column = divmod(index, size)
        mosaic[row * height:(row + 1) * height, column * width:(column + 1) * width] = as_image(sample, shape)

    fig, ax = plt.subplots(figsize=(8.5, 9))
    ax.imshow(mosaic, interpolation="nearest")
    ticks = np.arange(0, size, 2)
    ax.set_xticks(ticks * width + (width - 1) / 2, [f"{axis[i]:.1f}" for i in ticks])
    ax.set_yticks(ticks * height + (height - 1) / 2, [f"{axis[::-1][i]:.1f}" for i in ticks])
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig

def plot_kl_per_dimension(kl: Dict[str, npt.NDArray[Float]], title: str) -> Figure:
    """Mean KL of each latent dimension (sorted): a dimension near 0 is unused (q(z|x) = prior for every x)."""
    fig, axes = plt.subplots(1, len(kl), figsize=(4.6 * len(kl), 3.8), squeeze=False)
    for ax, (variant, values) in zip(axes[0], kl.items()):
        ordered = np.sort(values)[::-1]
        ax.bar(np.arange(1, len(ordered) + 1), ordered, color="tab:blue")
        ax.set_title(f"{variant}: {int(np.sum(ordered > 0.05))} of {len(ordered)} dimensions in use (KL > 0.05)", fontsize=9)
        ax.set_xlabel("Latent dimension (sorted)")
        ax.grid(alpha=0.3, axis="y")
    axes[0][0].set_ylabel("Mean KL per dimension")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig

def interpolation_titles(steps: int) -> List[str]:
    return [f"t={t:.2f}" for t in np.linspace(0, 1, steps)]
