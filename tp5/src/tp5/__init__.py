import argparse
from typing import Callable, Dict

from tp5.config import load_config
from tp5.functions.pixel_error import pixel_errors
from tp5.persistence.generated import GENERATED_DIR
from tp5.persistence.run_file import load_run, save_run
from tp5.runners.basic import run_basic
from tp5.types.config.autoencoder import AutoencoderType
from tp5.types.config.config import AutoencoderConfig
from tp5.types.training_run import TrainingRun

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

# What main runs for each autoencoder type of the config
RUNNERS: Dict[AutoencoderType, Callable[[AutoencoderConfig], TrainingRun]] = {
    AutoencoderType.BASIC: run_basic,
}

def report(run: TrainingRun) -> None:
    epoch, loss, max_errors, learned = run.history[-1]
    wrong = pixel_errors(run.model.reconstruct(run.inputs), run.inputs)

    print(f"Epochs: {epoch + 1} ({run.elapsed:.1f} s), seed {run.seed}")
    print(f"Final loss: {loss:.6g}")
    allowed = run.goal_pixels if run.goal_pixels is not None else 0
    print(f"Max wrong pixels: {max_errors}; patterns with ≤ {allowed} wrong pixels: {learned}/{len(run.inputs)}"
          + ("" if run.goal_pixels is not None else " (no goal)"))
    failed = [f"{label} ({count})" for label, count in zip(run.labels, wrong) if count > allowed]
    if failed:
        print("Not learned:", ", ".join(failed))

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
        serve(LatentExplorer.from_run(run, args.generated_dir), args.host, args.port, args.browser)
