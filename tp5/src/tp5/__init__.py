import argparse
from typing import Callable, Dict

from tp5.config import load_config
from tp5.functions.pixel_error import pixel_errors
from tp5.persistence.generated import GENERATED_DIR
from tp5.persistence.noisy_file import NOISY_DIR
from tp5.persistence.run_file import load_run, save_run
from tp5.runners.basic import run_basic
from tp5.models.denoising_autoencoder import DenoisingAutoencoder
from tp5.runners.denoising import run_denoising
from tp5.models.variational_autoencoder import VariationalAutoencoder
from tp5.runners.variational import run_variational
from tp5.types.config.autoencoder import AutoencoderType
from tp5.types.config.config import AutoencoderConfig
from tp5.types.training_run import TrainingRun

import numpy as np

parser = argparse.ArgumentParser(description="Autoencoders: basic, denoising and variational.")

parser.add_argument("-c", "--config", dest="config", type=str, required=False, help="Configuration JSON file (required unless --load)")
parser.add_argument("-l", "--load", dest="load", type=str, required=False, help="Load a trained run (.npz saved with --output) instead of training")
parser.add_argument("-o", "--output", dest="output", type=str, required=False, help="Output .npz file for the trained model and results (optional)")
parser.add_argument("-p", "--plot", dest="plot", action="store_true", help="Save the plots of the run as PNG into --plot-dir")
parser.add_argument("--plot-dir", dest="plot_dir", type=str, default="res/output/plots", help="Directory for the plots (default res/output/plots)")

parser.add_argument("-i", "--interactive", dest="interactive", action="store_true", help="Open the web latent space explorer with the trained model")
parser.add_argument("--host", dest="host", type=str, default="127.0.0.1", help="Host of the explorer (default 127.0.0.1)")
parser.add_argument("--port", dest="port", type=int, default=8000, help="Port of the explorer (default 8000)")
parser.add_argument("--no-browser", dest="browser", action="store_false", help="Do not open the browser automatically")
parser.add_argument("--generated-dir", dest="generated_dir", type=str, default=GENERATED_DIR, help=f"Where the explorer's 'Save image' writes (default {GENERATED_DIR})")
parser.add_argument("--noisy-dir", dest="noisy_dir", type=str, default=NOISY_DIR, help=f"Noisy pattern files listed and saved by the /denoise page (default {NOISY_DIR})")

# What main runs for each autoencoder type of the config
RUNNERS: Dict[AutoencoderType, Callable[[AutoencoderConfig], TrainingRun]] = {
    AutoencoderType.BASIC: run_basic,
    AutoencoderType.DENOISING: run_denoising,
    AutoencoderType.VARIATIONAL: run_variational,
}

def report(run: TrainingRun) -> None:
    epoch, loss, max_errors, learned = run.history[-1]
    reconstructions = run.model.reconstruct(run.inputs)
    wrong = pixel_errors(reconstructions, run.inputs)

    print(f"Epochs: {epoch + 1} ({run.elapsed:.1f} s), seed {run.seed}, {len(run.inputs)} samples {run.shape.height}x{run.shape.width}x{run.shape.channels}")

    if isinstance(run.model, VariationalAutoencoder):
        report_variational(run, loss, reconstructions)
        return
    if not np.isin(run.inputs, (0.0, 1.0)).all():
        # Continuous images (e.g. photos): wrong pixels mean nothing, mean absolute error instead
        errors = np.mean(np.abs(reconstructions - run.inputs), axis=1)
        print(f"Final loss: {loss:.6g}; reconstruction MAE mean {errors.mean():.4f}, max {errors.max():.4f}")
        return
    allowed = run.goal_pixels if run.goal_pixels is not None else 0
    goal = "" if run.goal_pixels is not None else " (no goal)"

    if isinstance(run.model, DenoisingAutoencoder):
        # The history measures denoising: a fixed noisy set (copies x patterns) against the clean patterns
        model = run.model
        evaluated = len(run.inputs) * model.copies
        print(f"Noise: {model.noise.__name__} level {model.noise_level}, {model.copies} copies per pattern")
        print(f"Denoising (fixed noisy set of {evaluated}): loss {loss:.6g}, max wrong pixels {max_errors}; "
              f"outputs with ≤ {allowed} wrong pixels: {learned}/{evaluated}{goal}")
        print(f"Clean inputs: max wrong pixels {int(wrong.max())}; patterns with ≤ {allowed} wrong pixels: "
              f"{int((wrong <= allowed).sum())}/{len(run.inputs)}")
    else:
        print(f"Final loss: {loss:.6g}")
        print(f"Max wrong pixels: {max_errors}; patterns with ≤ {allowed} wrong pixels: {learned}/{len(run.inputs)}{goal}")
    failed = [f"{label} ({count})" for label, count in zip(run.labels, wrong) if count > allowed]
    if failed:
        print("Not learned (clean input):", ", ".join(failed))

def report_variational(run: TrainingRun, loss: float, reconstructions) -> None:
    model = run.model
    assert isinstance(model, VariationalAutoencoder)
    reconstruction, kl = model.loss_terms[-1] if model.loss_terms else (float("nan"), float("nan"))
    print(f"β = {model.beta} (KL weight per sample {model.kl_weight:.4g}); "
          f"final J {loss:.6g} = reconstruction {reconstruction:.6g} + KL {kl:.4g} · {model.kl_weight:.4g}")
    errors = np.mean(np.abs(reconstructions - run.inputs), axis=1)
    print(f"Reconstruction of decode(μ): MAE mean {errors.mean():.4f}, max {errors.max():.4f}")
    if np.isin(run.inputs, (0.0, 1.0)).all():
        wrong = pixel_errors(reconstructions, run.inputs)
        print(f"Wrong pixels: max {int(wrong.max())}, patterns with 0 wrong: {int((wrong == 0).sum())}/{len(run.inputs)}")
    # Codes close to the prior N(0, I): mean ≈ 0 and spread ≈ 1; σ ≈ 1 with μ ≈ 0 = dimension unused
    mu, logvar = model.encode_distribution(run.inputs)
    sigma = np.exp(0.5 * logvar)
    print("Latent μ mean:", np.array2string(mu.mean(axis=0), precision=3))
    print("Latent μ std: ", np.array2string(mu.std(axis=0), precision=3))
    print("Mean σ(x):    ", np.array2string(sigma.mean(axis=0), precision=3))

def main():
    args = parser.parse_args()
    if (args.config is None) == (args.load is None):
        parser.error("pass either --config (train) or --load (trained run), not both")

    if args.load:
        run = load_run(args.load)
        print(f"Loaded {args.load}")
    else:
        config = load_config(args.config)

        autoencoder_type = config["autoencoder"]["type"]
        if autoencoder_type not in RUNNERS:
            raise NotImplementedError(f"The '{autoencoder_type.value}' autoencoder is not implemented yet.")

        run = RUNNERS[autoencoder_type](config)
    report(run)

    if args.output:
        save_run(args.output, run)
        print(f"Saved to {args.output}")

    if args.plot:
        # Imported only when plotting, so training alone never loads matplotlib. Non-interactive
        # backend: figures are saved to files, never shown in a window
        import matplotlib
        matplotlib.use("Agg")
        from tp5.visualization.autoencoder import plot_run
        plot_run(run, save_dir=args.plot_dir)
        print(f"Plots saved to {args.plot_dir}")

    if args.interactive:
        # Imported only when needed, like the plots
        from tp5.web.explorer import LatentExplorer
        from tp5.web.server import serve
        serve(LatentExplorer.from_run(run, args.generated_dir, args.noisy_dir), args.host, args.port, args.browser)
