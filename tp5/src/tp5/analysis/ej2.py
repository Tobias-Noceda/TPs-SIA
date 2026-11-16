import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tp5.experiments.ej2_sweep import STUDIES
from tp5.models.variational_autoencoder import VariationalAutoencoder
from tp5.persistence.run_file import load_run
from tp5.types.float import Float
from tp5.types.training_run import TrainingRun
from tp5.visualization.vae_sweep import (
    interpolation_titles, plot_grid, plot_kl_per_dimension, plot_latent_maps, plot_loss_terms, plot_mosaic, plot_rows,
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
    parser.add_argument("--pairs", nargs="+", default=["1F600-1F60E", "1F621-1F608", "1F62D-1F60D"], help="Interpolations, as label-label")
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
    shape = (int(first.shape[0]), int(first.shape[1]), int(first.shape[2]))

    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    def save(figure: Any, name: str) -> None:
        figure.savefig(plots / name, dpi=130)
        matplotlib.pyplot.close(figure)

    save(plot_mosaic(inputs, shape, f"Dataset: {len(inputs)} images of {shape[0]}x{shape[1]}x{shape[2]}"), "dataset.png")

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

        kl = {}
        for variant, run in variants.items():
            if isinstance(run.model, VariationalAutoencoder):
                mu, logvar = run.model.encode_distribution(inputs)
                kl[_name(study, variant)] = np.mean(0.5 * (np.exp(logvar) + mu ** 2 - 1 - logvar), axis=0)
        if study == "latent" and kl:
            save(plot_kl_per_dimension(kl, "latent: how many dimensions the VAE really uses"), "latent_kl_per_dimension.png")

    # 2c: generation with each reference model (figures suffixed with its name)
    for reference in args.reference:
        study, _, variant = reference.partition("__")
        if (study, variant) not in runs:
            print(f"No {reference} run: skipping its generation figures.")
            continue
        _generation(runs[(study, variant)], _name(study, variant), reference, args, parser, save)
    print(f"Saved summary.csv and plots/ to {root}")

def _generation(reference: TrainingRun, name: str, suffix: str, args: argparse.Namespace, parser: argparse.ArgumentParser, save: Any) -> None:
    """New samples from the prior next to their closest training image, interpolations and (2-D) the decoded latent plane."""
    model = reference.model
    inputs, labels = reference.inputs, reference.labels
    shape = (int(reference.shape[0]), int(reference.shape[1]), int(reference.shape[2]))

    z = RandomState(args.seed).standard_normal((args.samples, model.latent_size))
    generated = model.decode(z)
    closest, distance = _closest(generated, inputs)
    save(plot_rows(
        [("generated", generated), ("closest in\nthe dataset", inputs[closest])], shape,
        f"{name}: new samples from z ~ N(0, I) and the training image closest to each (title = mean absolute error)",
        [[f"z=({', '.join(f'{v:.1f}' for v in code[:2])}{'…' if model.latent_size > 2 else ''})" for code in z],
         [f"{labels[i]}\nMAE {d:.3f}" for i, d in zip(closest, distance)]],
    ), f"generated_{suffix}.png")
    # Reference of how different two real samples are: each training image against its closest other one
    between = np.mean(np.abs(inputs[:, None, :] - inputs[None, :, :]), axis=2)
    np.fill_diagonal(between, np.inf)
    print(f"\n{name} — generated samples: MAE to the closest training image {distance.mean():.3f} (min {distance.min():.3f}, max {distance.max():.3f}); "
          f"between training images the closest pair differs {between.min(axis=1).mean():.3f} on average")

    codes = model.encode(inputs)
    steps = 9
    t = np.linspace(0, 1, steps)[:, None]
    interpolations = []
    for pair in args.pairs:
        start, _, end = pair.partition("-")
        if start not in labels or end not in labels:
            parser.error(f"unknown label in {pair}; available: {' '.join(labels)}")
        a, b = codes[labels.index(start)], codes[labels.index(end)]
        interpolations.append((f"{start} → {end}", model.decode(a + t * (b - a))))
    save(plot_rows(interpolations, shape, f"{name}: decoded points between the codes of two training images",
                   [interpolation_titles(steps)] * len(interpolations)), f"interpolation_{suffix}.png")

    if model.latent_size == 2:
        axis = np.linspace(-2.5, 2.5, args.grid)
        grid = np.array([[a, b] for b in axis[::-1] for a in axis])
        save(plot_grid(model.decode(grid), shape, axis, f"{name}: decoder output over the latent plane (±2.5 σ of the prior)"), f"latent_grid_{suffix}.png")

if __name__ == "__main__":
    main()
