import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from tp5.interfaces.autoencoder import Autoencoder
from tp5.types.image_shape import ImageShape
from tp5.types.samples import Samples
from tp5.types.training_run import TrainingRun

import numpy as np

# Default folder of the images generated from the latent space (ex. 1a.4)
GENERATED_DIR = "res/output/ej1/generated"

# File names chosen by the user: letters, digits, spaces, '.', '_', '-', '+', '(', ')', ',' (no folders)
_NAME = re.compile(r"[\w .+\-(),]+")

def validate_name(name: str) -> str:
    """File name (without extension) for a generated image; raises ValueError if it is not a plain name."""
    name = name.strip().removesuffix(".png").strip()
    if not name or name in (".", "..") or not _NAME.fullmatch(name):
        raise ValueError("The name may only contain letters, digits, spaces and . _ - + ( ) , (no folders).")
    return name

def run_metadata(run: TrainingRun) -> Dict[str, Any]:
    """Seed, model parameters, cut conditions and training result of a run, to label what it generates."""
    source = run.source
    epoch, loss, max_errors, learned = run.history[-1] if run.history else (-1, np.nan, -1, 0)

    def layers(network: Dict[str, Any]) -> List[str]:
        described = []
        for layer in network["arquitecture"]:
            if isinstance(layer, dict):
                described.append(f"{layer['neurons']} {layer['activation_function']['name']}")
            else:
                names = sorted({neuron["activation_function"]["name"] for neuron in layer})
                described.append(f"{len(layer)} {'/'.join(names)}")
        return described

    encoder = layers(source["autoencoder"]["encoder"])
    decoder = layers(source["autoencoder"]["decoder"])
    sizes = [str(run.inputs.shape[1])] + [layer.split()[0] for layer in encoder + decoder]

    return {
        "seed": run.seed,
        "type": source["autoencoder"]["type"],
        "arquitecture": "-".join(sizes),
        "encoder": encoder,
        "decoder": decoder,
        "loss_function": source.get("loss_function", "mse"),
        "learning_rate": source["learning_rate"],
        "batch_size": source.get("batch_size", 1),
        "initialization": source.get("initialization", "random"),
        "cut_conditions": {
            "epochs": source["epochs"],
            "epsilon": source.get("epsilon", 1e-6),
            "goal_pixels": run.goal_pixels,
            "stop_at_goal": bool(source.get("stop_at_goal", True)) and run.goal_pixels is not None,
        },
        "training": {
            "epochs_trained": int(epoch) + 1,
            "final_loss": float(loss),
            "max_wrong_pixels": int(max_errors),
            "patterns_learned": int(learned),
            "patterns": len(run.inputs),
        },
        "dataset": source["dataset"]["path"],
    }

def save_generated(
    model: Autoencoder,
    z: Sequence[float],
    name: str,
    shape: ImageShape,
    metadata: Dict[str, Any],
    directory: str | Path = GENERATED_DIR,
    binarize: bool = True,
    ink: bool = True,
    training_inputs: Samples | None = None,
    labels: Sequence[str] | None = None,
) -> List[Path]:
    """
    Decode the latent point `z` and save it in `directory` as report-ready figures: the image next to
    the latent coordinates, model, seed, cut conditions and training result needed to replicate it.

    - `{name}.png`: the decoder output as is (grayscale levels / colours);
    - `{name}_binarized.png`: thresholded at 0.5, only when `binarize` (binary datasets such as the font).

    `name` is chosen by the user (see validate_name). Existing files are never overwritten: raises
    FileExistsError instead.

    Returns:
        Paths of the written files.
    """
    # Imported here: only saving needs matplotlib
    from tp5.visualization.generated import plot_generated

    point = np.asarray(z, dtype=np.float64).reshape(1, -1)
    if not np.isfinite(point).all():
        raise ValueError("Latent coordinates must be finite.")
    image = np.clip(model.decode(point)[0], 0, 1)

    coordinates = [float(v) for v in point[0]]
    info: Dict[str, Any] = {**metadata, "latent": coordinates}
    if training_inputs is not None:
        if binarize:
            distances = np.sum((image[None, :] >= 0.5) != (training_inputs >= 0.5), axis=1)
        else:
            distances = np.mean(np.abs(image[None, :] - training_inputs), axis=1)
        nearest = int(np.argmin(distances))
        info["nearest_sample"] = {
            "label": labels[nearest] if labels is not None else nearest,
            "distance": f"{int(distances[nearest])} wrong px" if binarize else f"MAE {distances[nearest]:.4f}",
        }

    name = validate_name(name)
    directory = Path(directory)

    figures = [(directory / f"{name}.png", image, False)]
    if binarize:
        figures.append((directory / f"{name}_binarized.png", (image >= 0.5).astype(np.float64), True))

    existing = [path.name for path, _, _ in figures if path.exists()]
    if existing:
        raise FileExistsError(f"Already exists: {', '.join(existing)}. Choose another name.")
    directory.mkdir(parents=True, exist_ok=True)

    for path, pixels, binarized in figures:
        plot_generated(pixels, shape, info, ink, binarized).savefig(path, dpi=150)
    return [path for path, _, _ in figures]
