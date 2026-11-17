import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tp5.experiments.ej2_sweep import STUDIES
from tp5.models.variational_autoencoder import VariationalAutoencoder
from tp5.persistence.run_file import load_run
from tp5.types.float import Float
from tp5.types.training_run import TrainingRun
from tp5.visualization.vae_sweep import (
    interpolation_titles, plot_class_maps, plot_grid, plot_kl_per_dimension, plot_latent_maps, plot_loss_terms, plot_mosaic,
    plot_rows,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib
from numpy.random import RandomState

# Ex. 2: what the KL term does to the latent space (2b) and what the trained model generates (2c).
# Every model decodes the same z ~ N(0, I) (same seed), so the comparisons between variants are paired.

# Name the plots give to each variant of the beta study
def _name(study: str, variant: str) -> str:
    if study == "latent":
        return f"latent {variant}"
    return "plain autoencoder (no KL)" if variant == "ae" else f"VAE β = {variant}"

def _closest(outputs: npt.NDArray[Float], inputs: npt.NDArray[Float]) -> Tuple[npt.NDArray[np.intp], npt.NDArray[Float]]:
    """Index of the training sample closest to each output (mean absolute error) and that error."""
    errors = np.mean(np.abs(outputs[:, None, :] - inputs[None, :, :]), axis=2)
    closest = errors.argmin(axis=1)
    return closest, errors[np.arange(len(outputs)), closest]

def _class_of(label: str) -> str:
    """
    Class of a sample: its folder (homer_simpson/pic_0012 → homer_simpson) or, in a flat folder, the
    file name without a numeric suffix (homer_simpson_03 → homer_simpson); otherwise the label itself.
    """
    if "/" in label:
        return label.split("/", 1)[0]
    return re.sub(r"_\d+$", "", label)

def _nearest_real(inputs: npt.NDArray[Float], classes: List[str], same_class: bool | None, chunk: int = 32) -> npt.NDArray[Float]:
    """
    MAE of every training image to its closest other training image: of the same class (True), of
    another class (False) or any (None). Computed in chunks: all the pairs at once would need
    samples² · pixels floats (≈ 2 GB for 300 images of 32x32x3).
    """
    labels = np.array(classes)
    distances = np.full(len(inputs), np.inf)
    for start in range(0, len(inputs), chunk):
        rows = slice(start, start + chunk)
        errors = np.mean(np.abs(inputs[rows, None, :] - inputs[None, :, :]), axis=2)
        indices = np.arange(start, start + len(errors))
        errors[np.arange(len(errors)), indices] = np.inf
        if same_class is not None:
            same = labels[indices][:, None] == labels[None, :]
            errors[~same if same_class else same] = np.inf
        distances[rows] = errors.min(axis=1)
    return distances

def _short_names(labels: List[str]) -> List[str]:
    """
    Labels without the prefix and suffix all of them share, cut at "_" (argentina_boca-juniors_64x64.football-logos.cc
    → boca-juniors): long file names do not fit as plot titles. Labels without a shared part stay as they are.
    """
    if len(labels) < 2:
        return list(labels)
    prefix = os.path.commonprefix(labels)
    prefix = prefix[:prefix.rfind("_") + 1]
    suffix = os.path.commonprefix([label[::-1] for label in labels])[::-1]
    suffix = suffix[suffix.find("_"):] if "_" in suffix else ""
    short = [label[len(prefix):len(label) - len(suffix)] for label in labels]
    return short if all(short) and len(set(short)) == len(short) else list(labels)

def _endpoint(name: str, codes: npt.NDArray[Float], labels: List[str]) -> npt.NDArray[Float] | None:
    """Code of an interpolation endpoint: a sample label (full or short), or a class name (mean code of its samples)."""
    if name in labels:
        return codes[labels.index(name)]
    short = _short_names(labels)
    if name in short:
        return codes[short.index(name)]
    members = [i for i, label in enumerate(labels) if _class_of(label) == name]
    return codes[members].mean(axis=0) if members else None

def _sigma(run: TrainingRun) -> npt.NDArray[Float] | None:
    if not isinstance(run.model, VariationalAutoencoder):
        return None
    return np.exp(0.5 * run.model.encode_distribution(run.inputs)[1])

def main() -> None:
    parser = argparse.ArgumentParser(description="Ex. 2: latent space and generated samples of the VAE sweep.")
    parser.add_argument("-i", "--input", default="res/output/ej2", help="Sweep output directory (with runs/)")
    parser.add_argument("-r", "--reference", nargs="+", default=["beta__1", "latent__8"],
                        help="study__variant of the models used to generate (default beta__1 latent__8)")
    parser.add_argument("-n", "--samples", type=int, default=12, help="Samples drawn from the prior per model")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the z ~ N(0, I) every model decodes")
    parser.add_argument("--pairs", nargs="+", default=["homer_simpson-marge_simpson", "bart_simpson-lisa_simpson", "ned_flanders-mr_burns"],
                        help="Interpolations, as a-b: sample labels (file names) or classes (folders: mean code of their samples); "
                             "with emojis24 e.g. 1F600-1F60E")
    parser.add_argument("--grid", type=int, default=13, help="Side of the decoded grid of the latent plane")
    args = parser.parse_args()
    matplotlib.use("Agg")

    root = Path(args.input)
    paths = list((root / "runs").glob("*__*__s*.npz"))
    if not paths:
        raise SystemExit(f"No runs found in {root / 'runs'}")
    # Studies and variants in the order the sweep defines them; only the first seed is drawn
    order = [f"{study}__{variant}" for study, variants in STUDIES.items() for variant in variants]
    paths.sort(key=lambda path: (order.index(name) if (name := path.stem.rsplit("__", 1)[0]) in order else len(order), path.stem))

    runs: Dict[Tuple[str, str], TrainingRun] = {}
    rows: List[Dict[str, Any]] = []
    for path in paths:
        study, variant, seed = path.stem.split("__")
        run = load_run(str(path))
        runs.setdefault((study, variant), run)
        model = run.model
        reconstructions = model.reconstruct(run.inputs)
        codes = model.encode(run.inputs)
        sigma = _sigma(run)
        row = {
            "study": study, "variant": variant, "seed": int(seed.removeprefix("s")), "epochs": len(run.history),
            "latent": codes.shape[1], "loss": float(run.history[-1][1]),
            "reconstruction_mae": float(np.mean(np.abs(reconstructions - run.inputs))),
            "code_mean": float(np.abs(codes.mean(axis=0)).mean()), "code_std": float(codes.std(axis=0).mean()),
            "sigma_mean": float(sigma.mean()) if sigma is not None else np.nan,
            "reconstruction": np.nan, "kl": np.nan, "elapsed": run.elapsed,
        }
        if isinstance(model, VariationalAutoencoder) and model.loss_terms:
            row["reconstruction"], row["kl"] = model.loss_terms[-1]
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "summary.csv", index=False)
    print(summary.drop(columns=["seed", "epochs", "elapsed"]).round(4).to_string(index=False))

    first = next(iter(runs.values()))
    inputs, labels = first.inputs, first.labels
    classes = [_class_of(label) for label in labels]
    # Several samples per class (subfolders): the class figures and references make sense
    variants_per_class = len(set(classes)) < len(classes)
    shape = (int(first.shape[0]), int(first.shape[1]), int(first.shape[2]))

    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    def save(figure: Any, name: str) -> None:
        figure.savefig(plots / name, dpi=130)
        matplotlib.pyplot.close(figure)

    described = f"Dataset: {len(inputs)} images of {shape[0]}x{shape[1]}x{shape[2]}"
    if variants_per_class:
        described += f" ({len(set(classes))} classes)"
    save(plot_mosaic(inputs, shape, described, columns=20 if len(inputs) > 160 else 16), "dataset.png")

    # Samples the reconstruction rows show: evenly spread over the dataset
    shown = np.linspace(0, len(inputs) - 1, args.samples).astype(int)
    for study in dict.fromkeys(study for study, _ in runs):
        variants = {variant: run for (s, variant), run in runs.items() if s == study}

        curves = {}
        for variant, run in variants.items():
            model = run.model
            if isinstance(model, VariationalAutoencoder) and model.loss_terms:
                terms = np.array(model.loss_terms)
                curves[_name(study, variant)] = (terms[:, 0], terms[:, 1])
            else:
                curves[_name(study, variant)] = (np.array([loss for _, loss, _, _ in run.history], dtype=np.float64), None)
        save(plot_loss_terms(study, curves), f"{study}_loss.png")

        reconstructions = [("original", inputs[shown])] + [
            (_name(study, variant), run.model.reconstruct(inputs[shown])) for variant, run in variants.items()
        ]
        save(plot_rows(reconstructions, shape, f"{study}: reconstructions, decode(μ(x))"), f"{study}_reconstructions.png")

        # Generation: every model decodes z ~ N(0, I); the first dimensions are shared between sizes
        z = RandomState(args.seed).standard_normal((args.samples, max(run.model.latent_size for run in variants.values())))
        generated = [(_name(study, variant), run.model.decode(z[:, :run.model.latent_size])) for variant, run in variants.items()]
        save(plot_rows(generated, shape, f"{study}: new samples, decode(z) with z ~ N(0, I)"), f"{study}_samples.png")

        flat = {_name(study, variant): (run.model.encode(inputs), _sigma(run)) for variant, run in variants.items() if run.model.latent_size == 2}
        if flat:
            save(plot_latent_maps(flat, inputs, shape, f"{study}: training samples in the latent space; circles = 1, 2, 3 σ of the prior N(0, I), "
                                  "ellipses = 1 σ(x)"), f"{study}_latent.png")
            if variants_per_class:
                save(plot_class_maps({variant: codes for variant, (codes, _) in flat.items()}, classes,
                                     f"{study}: codes colored by class, name at the centroid of each class"), f"{study}_classes.png")

        kl = {}
        for variant, run in variants.items():
            if isinstance(run.model, VariationalAutoencoder):
                mu, logvar = run.model.encode_distribution(inputs)
                kl[_name(study, variant)] = np.mean(0.5 * (np.exp(logvar) + mu ** 2 - 1 - logvar), axis=0)
        if study == "latent" and kl:
            save(plot_kl_per_dimension(kl, "latent: how many dimensions the VAE really uses"), "latent_kl_per_dimension.png")

    # Reference of how different two real samples are, for judging the generated ones
    real = {"any": float(_nearest_real(inputs, classes, None).mean())}
    if variants_per_class:
        real["same_class"] = float(_nearest_real(inputs, classes, True).mean())
        real["other_class"] = float(_nearest_real(inputs, classes, False).mean())
    print("Closest other training image (MAE, mean over the dataset): "
          + ", ".join(f"{key.replace('_', ' ')} {value:.3f}" for key, value in real.items()))

    # 2c: generation with each reference model (figures suffixed with its name)
    generation_rows: List[Dict[str, Any]] = []
    for reference in args.reference:
        study, _, variant = reference.partition("__")
        if (study, variant) not in runs:
            print(f"No {reference} run: skipping its generation figures.")
            continue
        generation_rows.append(_generation(runs[(study, variant)], _name(study, variant), reference, real, args, parser, save))
    if generation_rows:
        pd.DataFrame(generation_rows).to_csv(root / "generation.csv", index=False)
    print(f"Saved summary.csv, generation.csv and plots/ to {root}")

def _generation(reference: TrainingRun, name: str, suffix: str, real: Dict[str, float], args: argparse.Namespace,
                parser: argparse.ArgumentParser, save: Any) -> Dict[str, Any]:
    """New samples from the prior next to their closest training image, interpolations and (2-D) the decoded latent plane."""
    model = reference.model
    inputs, labels = reference.inputs, reference.labels
    shape = (int(reference.shape[0]), int(reference.shape[1]), int(reference.shape[2]))

    short = _short_names(labels)
    z = RandomState(args.seed).standard_normal((args.samples, model.latent_size))
    generated = model.decode(z)
    closest, distance = _closest(generated, inputs)
    save(plot_rows(
        [("generated", generated), ("closest in\nthe dataset", inputs[closest])], shape,
        f"{name}: new samples from z ~ N(0, I) and the training image closest to each (title = mean absolute error)",
        [[f"z=({', '.join(f'{v:.1f}' for v in code[:2])}{'…' if model.latent_size > 2 else ''})" for code in z],
         [f"{_class_of(short[i])}\nMAE {d:.3f}" for i, d in zip(closest, distance)]],
    ), f"generated_{suffix}.png")
    # A generated sample much closer to a training image than real images are to each other is a copy;
    # about as close as two variants of the same class, a new variant of that class
    reference_text = ", ".join(f"{key.replace('_', ' ')} {value:.3f}" for key, value in real.items())
    print(f"\n{name} — generated samples: MAE to the closest training image {distance.mean():.3f} (min {distance.min():.3f}, "
          f"max {distance.max():.3f}); closest other real image: {reference_text}")

    codes = model.encode(inputs)
    steps = 9
    t = np.linspace(0, 1, steps)[:, None]
    interpolations = []
    for pair in args.pairs:
        # Labels can contain "-" themselves (boca-juniors): the first cut where both sides are known wins
        cuts = [(pair[:i], pair[i + 1:]) for i, char in enumerate(pair) if char == "-"]
        start, end = next(((a, b) for a, b in cuts if _endpoint(a, codes, labels) is not None and _endpoint(b, codes, labels) is not None),
                          cuts[0] if cuts else (pair, ""))
        a, b = _endpoint(start, codes, labels), _endpoint(end, codes, labels)
        if a is None or b is None:
            parser.error(f"unknown label or class in {pair}; available: {' '.join(dict.fromkeys(_class_of(label) for label in _short_names(labels)))}")
        interpolations.append((f"{start} → {end}", model.decode(a + t * (b - a))))
    save(plot_rows(interpolations, shape, f"{name}: decoded points between the codes of two training images",
                   [interpolation_titles(steps)] * len(interpolations)), f"interpolation_{suffix}.png")

    if model.latent_size == 2:
        axis = np.linspace(-2.5, 2.5, args.grid)
        grid = np.array([[a, b] for b in axis[::-1] for a in axis])
        save(plot_grid(model.decode(grid), shape, axis, f"{name}: decoder output over the latent plane (±2.5 σ of the prior)"), f"latent_grid_{suffix}.png")

    return {"reference": suffix, "generated_mae_mean": float(distance.mean()), "generated_mae_min": float(distance.min()),
            "generated_mae_max": float(distance.max()), **{f"real_{key}": value for key, value in real.items()}}

if __name__ == "__main__":
    main()
