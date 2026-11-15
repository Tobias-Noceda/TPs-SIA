from typing import Any, Dict, List

from tp5.types.image_shape import ImageShape

import numpy as np
import numpy.typing as npt
# Object-oriented API only (no pyplot): no GUI backend involved, safe to use from the server threads
from matplotlib.figure import Figure

def _format_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)

def describe(info: Dict[str, Any]) -> List[str]:
    """Lines of text with everything needed to replicate a generated image."""
    coordinates = ", ".join(f"{v:+.4f}" for v in info["latent"])
    lines = [
        "Latent point",
        f"  z = ({coordinates})",
    ]
    nearest = info.get("nearest_sample")
    if nearest is not None:
        lines.append(f"  nearest sample: '{nearest['label']}' ({nearest['distance']})")
    return lines + [""] + describe_model(info)

def describe_model(info: Dict[str, Any]) -> List[str]:
    """Model, seed, cut conditions, training result and dataset of a run (shared by every saved figure)."""
    learning_rate = info["learning_rate"]
    hyper = ", ".join(f"{k}={_format_number(v)}" for k, v in learning_rate.get("hyperparameters", {}).items())
    optimizer = f"{learning_rate['type']} (η = {_format_number(learning_rate['value'])}{', ' + hyper if hyper else ''})"
    cut = info["cut_conditions"]
    training = info["training"]
    goal = "none" if cut["goal_pixels"] is None else f"≤ {cut['goal_pixels']} px"

    lines = [
        "Model",
        f"  type: {info['type']}",
        f"  architecture: {info['arquitecture']}",
        f"  encoder: {', '.join(info['encoder'])}",
        f"  decoder: {', '.join(info['decoder'])}",
        f"  loss: {info['loss_function']}",
        f"  optimizer: {optimizer}",
        f"  batch size: {info['batch_size']}",
        f"  initialization: {info['initialization']}",
        f"  seed: {info['seed']}",
    ]
    noise = info.get("noise")
    if noise:
        lines.append(f"  training noise: {noise['type']} {_format_number(noise['level'])}, {noise['copies']} copies/epoch")
    lines += [
        "",
        "Cut conditions",
        f"  max epochs: {cut['epochs']}",
        f"  epsilon (loss): {_format_number(cut['epsilon'])}",
        f"  goal: {goal}, stop at goal: {'yes' if cut['stop_at_goal'] else 'no'}",
        "",
        "Training result",
        f"  epochs trained: {training['epochs_trained']}",
        f"  final loss: {_format_number(training['final_loss'])}",
        f"  max wrong pixels: {training['max_wrong_pixels']}",
        (f"  noisy outputs learned: {training['patterns_learned']}/{training['patterns']}"
         if training.get("noisy_evaluation") else
         f"  patterns learned: {training['patterns_learned']}/{training['patterns']}"),
        "",
        f"Dataset: {info['dataset']}",
    ]
    return lines

def plot_generated(image: npt.NDArray[np.float64], shape: ImageShape, info: Dict[str, Any], ink: bool, binarized: bool) -> Figure:
    """
    Figure of one decoded image (left, with pixel axes and a value scale) and the data to replicate it
    (right): latent coordinates, model, seed, cut conditions and training result.
    """
    values = np.clip(image, 0, 1).reshape(shape.height, shape.width, shape.channels)

    figure = Figure(figsize=(11.5, 5.6), layout="constrained")
    # Horizontal gap between the image (with its colour bar) and the metadata, as a fraction of the figure
    layout = figure.get_layout_engine()
    if (layout is not None):
        layout.set(wspace=0.1) # type: ignore
    image_ax, text_ax = figure.subplots(1, 2, width_ratios=[1, 1.35])

    if shape.channels == 1:
        shown = image_ax.imshow(values[:, :, 0], cmap="gray_r" if ink else "gray", vmin=0, vmax=1, interpolation="nearest")
        colorbar = figure.colorbar(shown, ax=image_ax, shrink=0.8)
        colorbar.set_label("pixel value" + (" (1 = ink)" if ink else ""))
    else:
        image_ax.imshow(values, interpolation="nearest")

    # Pixel grid
    image_ax.set_xticks(np.arange(shape.width))
    image_ax.set_yticks(np.arange(shape.height))
    image_ax.set_xticks(np.arange(-0.5, shape.width), minor=True)
    image_ax.set_yticks(np.arange(-0.5, shape.height), minor=True)
    image_ax.grid(which="minor", color="tab:gray", linewidth=0.5, alpha=0.6)
    image_ax.tick_params(which="minor", length=0)
    image_ax.set_xlabel("column")
    image_ax.set_ylabel("row")
    image_ax.set_title("Decoder output, binarized (≥ 0.5)" if binarized else "Decoder output")

    text_ax.axis("off")
    # Starts a bit inside its axes: extra margin from the colour bar labels
    text_ax.text(0.08, 1, "\n".join(describe(info)), va="top", ha="left", family="monospace", fontsize=9, transform=text_ax.transAxes)

    coordinates = ", ".join(f"{v:.3f}" for v in info["latent"])
    figure.suptitle(f"Generated sample at z = ({coordinates})", fontsize=13)
    return figure
