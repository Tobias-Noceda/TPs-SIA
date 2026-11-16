import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from tp5.experiments.ej1b_sweep import STUDIES
from tp5.functions.pixel_error import pixel_errors
from tp5.interfaces.autoencoder import Autoencoder
from tp5.persistence.run_file import load_run
from tp5.types.config.noise import NoiseFunctions
from tp5.types.float import Float
from tp5.types.noise_function import NoiseFunction
from tp5.types.samples import Samples
from tp5.visualization.denoising_sweep import (
    plot_confusion, plot_denoising_curves, plot_examples, plot_level_heatmap, plot_noise_types,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib
from numpy.random import RandomState

# Ex. 1b.2: every model of the sweep cleans the same corrupted patterns (same seed => same noise for
# all of them, so the comparison is paired) at several noise levels.

# Noise levels of the inputs each noise type is evaluated at
TEST_LEVELS: Dict[str, Tuple[float, ...]] = {
    "flip": (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4),
    "salt_pepper": (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8),
    "gaussian": (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8),
}

# Model the examples and the confusion matrix are drawn with
REFERENCE = ("train_level", "flip_0.1")

def corrupt(inputs: Samples, noise: NoiseFunction, level: float, draws: int, seed: int) -> Tuple[Samples, Samples]:
    """`draws` noisy versions of every pattern and the clean pattern of each row."""
    clean = np.tile(inputs, (draws, 1))
    return noise(RandomState(seed), clean, level), clean

def evaluate(model: Autoencoder, inputs: Samples, noise: NoiseFunction, level: float, draws: int, seed: int, tolerance: int) -> Tuple[float, float, float]:
    """(mean wrong pixels of the noisy input, of the output, fraction of outputs with ≤ tolerance wrong pixels)."""
    noisy, clean = corrupt(inputs, noise, level, draws, seed)
    errors = pixel_errors(model.reconstruct(noisy), clean)
    return float(pixel_errors(noisy, clean).mean()), float(errors.mean()), float(np.mean(errors <= tolerance))

def confusion_matrix(models: Sequence[Autoencoder], inputs: Samples, noise: NoiseFunction, level: float, draws: int, seed: int) -> npt.NDArray[Float]:
    """Fraction of the corruptions of each pattern (row) whose output is closest to each training pattern (column)."""
    count = len(inputs)
    matrix = np.zeros((count, count))
    noisy, _ = corrupt(inputs, noise, level, draws, seed)
    truth = np.tile(np.arange(count), draws)
    for model in models:
        outputs = model.reconstruct(noisy) >= 0.5
        closest = np.argmin(np.sum(outputs[:, None, :] != (inputs >= 0.5)[None, :, :], axis=2), axis=1)
        np.add.at(matrix, (truth, closest), 1)
    return matrix / matrix.sum(axis=1, keepdims=True)

def main() -> None:
    parser = argparse.ArgumentParser(description="Ex. 1b.2: how much noise the denoising autoencoders of the sweep remove.")
    parser.add_argument("-i", "--input", default="res/output/ej1b", help="Sweep output directory (with runs/)")
    parser.add_argument("-d", "--draws", type=int, default=50, help="Noisy versions of every pattern per noise level")
    parser.add_argument("-t", "--tolerance", type=int, default=1, help="Wrong pixels allowed for an output to count as recovered")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the evaluation noise")
    parser.add_argument("--examples", nargs="+", default=["a", "e", "g", "k", "s", "x"], help="Patterns drawn in the examples figure")
    parser.add_argument("--example-levels", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3], help="Flip levels of the examples figure")
    parser.add_argument("--confusion-level", type=float, default=0.2, help="Flip level of the confusion matrix")
    args = parser.parse_args()
    matplotlib.use("Agg")

    root = Path(args.input)
    paths = list((root / "runs").glob("*__*__s*.npz"))
    if not paths:
        raise SystemExit(f"No runs found in {root / 'runs'}")
    # Studies and variants in the order the sweep defines them (file names would sort 10 before 2)
    order = [f"{study}__{variant}" for study, variants in STUDIES.items() for variant in variants]
    paths.sort(key=lambda path: (order.index(name) if (name := path.stem.rsplit("__", 1)[0]) in order else len(order), path.stem))

    rows: List[Dict[str, Any]] = []
    models: Dict[Tuple[str, str], List[Autoencoder]] = {}
    for path in paths:
        study, variant, seed = path.stem.split("__")
        run = load_run(str(path))
        models.setdefault((study, variant), []).append(run.model)
        # Flip is the common axis of every study; the noise_type study is evaluated with the three noises
        for kind in (TEST_LEVELS if study == "noise_type" else ("flip",)):
            noise = NoiseFunctions(kind).function()
            for level in TEST_LEVELS[kind]:
                input_pixels, output_pixels, recovered = evaluate(run.model, run.inputs, noise, level, args.draws, args.seed, args.tolerance)
                rows.append({
                    "study": study, "variant": variant, "seed": int(seed.removeprefix("s")), "test_type": kind, "test_level": level,
                    "input_pixels": input_pixels, "output_pixels": output_pixels, "recovered": recovered,
                })
    inputs, labels, shape = run.inputs, run.labels, (int(run.shape[0]), int(run.shape[1]))

    summary = pd.DataFrame(rows)
    summary.to_csv(root / "summary.csv", index=False)
    aggregate = summary.groupby(["study", "variant", "test_type", "test_level"], sort=False).agg(
        runs=("seed", "count"),
        input_pixels=("input_pixels", "mean"),
        output_pixels=("output_pixels", "mean"),
        recovered=("recovered", "mean"),
    ).reset_index()
    aggregate.to_csv(root / "aggregate.csv", index=False)
    flip = aggregate[aggregate["test_type"] == "flip"]
    print(flip.pivot_table(index=["study", "variant"], columns="test_level", values="output_pixels", sort=False).round(2).to_string())

    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    def save(figure: Any, name: str) -> None:
        figure.savefig(plots / name, dpi=130)
        matplotlib.pyplot.close(figure)

    flips = summary[summary["test_type"] == "flip"]
    for study in summary["study"].unique():
        save(plot_denoising_curves(study, flips[flips["study"] == study], "flip", args.tolerance), f"{study}_curves.png")
    if "train_level" in set(summary["study"]):
        save(plot_level_heatmap(flips[flips["study"] == "train_level"], "flip"), "train_level_heatmap.png")
    if "noise_type" in set(summary["study"]):
        save(plot_noise_types(summary[summary["study"] == "noise_type"]), "noise_type_curves.png")

    if REFERENCE not in models:
        print(f"No {'__'.join(REFERENCE)} runs: skipping the examples and the confusion matrix.")
        return
    reference = models[REFERENCE]
    flip_noise = NoiseFunctions("flip").function()
    name = f"{REFERENCE[1]} (trained with), flip noise in the inputs"

    unknown = [label for label in args.examples if label not in labels]
    if unknown:
        parser.error(f"unknown labels {unknown}; available: {' '.join(labels)}")
    originals = inputs[[labels.index(label) for label in args.examples]]
    examples = []
    for level in args.example_levels:
        noisy, _ = corrupt(originals, flip_noise, level, 1, args.seed)
        examples.append((level, noisy, reference[0].reconstruct(noisy)))
    save(plot_examples(originals, args.examples, shape, examples, f"Denoising examples — {name}"), "examples.png")

    confusion = confusion_matrix(reference, inputs, flip_noise, args.confusion_level, args.draws, args.seed)
    save(plot_confusion(confusion, labels, f"Flip {args.confusion_level:g} in the inputs, model {REFERENCE[1]}: "
                        f"{100 * np.trace(confusion) / len(labels):.1f}% of the outputs are closest to the right pattern"), "confusion.png")
    worst = np.argsort(np.diag(confusion))[:5]
    print("Hardest patterns:", ", ".join(f"{labels[i]} ({100 * confusion[i, i]:.0f}%)" for i in worst))
    print(f"Saved summary.csv, aggregate.csv and plots/ to {root}")

if __name__ == "__main__":
    main()
