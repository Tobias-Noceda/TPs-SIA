from typing import Any, Dict

from tp5.types.image_shape import ImageShape
from tp5.visualization.generated import describe_model

import numpy as np
import numpy.typing as npt
# Object-oriented API only (no pyplot): safe to use from the server threads
from matplotlib.figure import Figure

def _show(ax, image: npt.NDArray[np.float64], shape: ImageShape, ink: bool, title: str) -> Any:
    values = np.clip(image, 0, 1).reshape(shape.height, shape.width, shape.channels)
    if shape.channels == 1:
        shown = ax.imshow(values[:, :, 0], cmap="gray_r" if ink else "gray", vmin=0, vmax=1, interpolation="nearest")
    else:
        shown = ax.imshow(values, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, shape.width), minor=True)
    ax.set_yticks(np.arange(-0.5, shape.height), minor=True)
    ax.grid(which="minor", color="tab:gray", linewidth=0.5, alpha=0.6)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    ax.set_title(title, fontsize=10)
    return shown

def _describe_noise(noise: Dict[str, Any]) -> str:
    """Noise type, level and seed of the input, plus where it came from (noisy file, edited by hand)."""
    text = str(noise.get("type", "manual"))
    if noise.get("level") is not None:
        text += f" {float(noise['level']):g}"
    if noise.get("seed") is not None:
        text += f", seed {noise['seed']}"
    if noise.get("variant") is not None:
        text += f", variant {noise['variant']}"
    if noise.get("file"):
        text += f" (file {noise['file']})"
    if noise.get("edited"):
        text += ", edited by hand"
    return text

def plot_denoising(
    original: npt.NDArray[np.float64],
    noisy: npt.NDArray[np.float64],
    output: npt.NDArray[np.float64],
    shape: ImageShape,
    info: Dict[str, Any],
    ink: bool,
    binary: bool,
) -> Figure:
    """
    Original, noisy input and decoder output (+ binarized output for binary datasets) next to the data
    to replicate it: sample, input noise and its errors, model, seed, cut conditions and training result.
    `info` = run metadata + `sample`, `input_noise`, `errors` (see LatentExplorer.save_denoising).
    """
    panels = 4 if binary else 3
    figure = Figure(figsize=(3 * panels + 6, 5.6), layout="constrained")
    figure.get_layout_engine().set(wspace=0.06)
    axes = figure.subplots(1, panels + 1, width_ratios=[1] * panels + [2.6])
    *image_axes, text_ax = axes

    errors = info["errors"]
    unit = "wrong px" if binary else "MAE"
    _show(image_axes[0], original, shape, ink, f"Original '{info['sample']}'")
    _show(image_axes[1], noisy, shape, ink, f"Noisy input\n{errors['noisy']} {unit}")
    shown = _show(image_axes[2], output, shape, ink, f"Decoder output\n{errors['output']} {unit}")
    if binary:
        _show(image_axes[3], (output >= 0.5).astype(np.float64), shape, ink, f"Output binarized (≥ 0.5)\n{errors['output']} {unit}")
    if shape.channels == 1:
        colorbar = figure.colorbar(shown, ax=image_axes, shrink=0.6, location="bottom")
        colorbar.set_label("pixel value" + (" (1 = ink)" if ink else ""))

    noise = info["input_noise"]
    lines = [
        f"Denoising '{info['sample']}'",
        "",
        "Input",
        f"  sample: '{info['sample']}' (#{info['index']})",
        "  noise: " + _describe_noise(noise),
        f"  noisy vs original: {errors['noisy']} {unit}",
        f"  output vs original: {errors['output']} {unit}",
        f"  output nearest sample: '{info['nearest']['label']}' ({info['nearest']['distance']})",
        "",
    ] + describe_model(info)

    text_ax.axis("off")
    text = text_ax.text(0.04, 1, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9, transform=text_ax.transAxes)
    # Left out of the constrained layout: its height made the layout push the panel titles out of the figure
    text.set_in_layout(False)
    return figure
