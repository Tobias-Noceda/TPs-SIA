import argparse
import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from tp5.config import parse_config
from tp5.persistence.run_file import save_run
from tp5.runners.denoising import run_denoising

# Study of ex. 1b: denoising autoencoders that differ in a single aspect of the base config (the
# noise they are trained with, the size of the latent space). It only trains and saves the models:
# analysis/ej1b.py corrupts the patterns at several levels and measures how much noise each removes.

type Variant = Callable[[Dict[str, Any]], None]

def noise(kind: str, level: float) -> Variant:
    def apply(config: Dict[str, Any]) -> None:
        config["noise"].update(type=kind, level=level)
    return apply

def latent(size: int) -> Variant:
    """Same hidden layers, another bottleneck."""
    def apply(config: Dict[str, Any]) -> None:
        config["autoencoder"]["encoder"]["arquitecture"][-1]["neurons"] = size
    return apply

STUDIES: Dict[str, Dict[str, Variant]] = {
    # Level 0 is a basic autoencoder with the same architecture: the baseline
    "train_level": {
        "flip_0": noise("flip", 0.0),
        "flip_0.05": noise("flip", 0.05),
        "flip_0.1": noise("flip", 0.1),
        "flip_0.2": noise("flip", 0.2),
        "flip_0.3": noise("flip", 0.3),
    },
    "latent": {
        "2": latent(2),
        "5": latent(5),
        "10": latent(10),
        "20": latent(20),
    },
    # Levels with a similar amount of damage: salt & pepper only changes half of the pixels it picks
    "noise_type": {
        "flip_0.1": noise("flip", 0.1),
        "salt_pepper_0.2": noise("salt_pepper", 0.2),
        "gaussian_0.4": noise("gaussian", 0.4),
    },
}

def _run(config: Dict[str, Any], path: str) -> str:
    # Workers run silently: one progress bar per process would interleave
    os.environ["TQDM_DISABLE"] = "1"
    try:
        save_run(path, run_denoising(parse_config(config)))
        return f"done {Path(path).name}"
    except FloatingPointError as error:
        return f"diverged {Path(path).name}: {error}"

def main() -> None:
    parser = argparse.ArgumentParser(description="Ex. 1b: sweep of denoising autoencoders (training noise, latent size, noise type).")
    parser.add_argument("-c", "--config", default="res/config/ej1b/denoising.json", help="Base config (compact layers, with a noise block)")
    parser.add_argument("-o", "--output", default="res/output/ej1b", help="Output directory (runs/ is created inside)")
    parser.add_argument("-s", "--seeds", type=int, default=3, help="Seeds per variant (0 … seeds - 1)")
    parser.add_argument("-e", "--epochs", type=int, default=8000, help="Epochs of every run")
    parser.add_argument("--studies", nargs="*", default=list(STUDIES), choices=list(STUDIES), help="Studies to run")
    parser.add_argument("-w", "--workers", type=int, default=os.cpu_count(), help="Parallel processes")
    parser.add_argument("--force", action="store_true", help="Re-run variants whose output already exists")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        base = json.load(file)
    # Every variant trains the same number of epochs: with noise there is no pixel goal that all of
    # them can reach, and the loss on each one's own noisy set is not comparable between variants
    base.update(epochs=args.epochs, epsilon=0.0)
    for name in ("goal_pixels", "stop_at_goal"):
        base.pop(name, None)
    # Validates the base before launching anything
    parse_config(base, args.config)

    runs_dir = Path(args.output) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    jobs: List[Tuple[Dict[str, Any], str]] = []
    for study in args.studies:
        for variant, apply in STUDIES[study].items():
            for seed in range(args.seeds):
                # The analysis groups the runs by this name: study__variant__s{seed}
                path = runs_dir / f"{study}__{variant}__s{seed}.npz"
                if path.exists() and not args.force:
                    continue
                config = copy.deepcopy(base)
                apply(config)
                config["seed"] = seed
                jobs.append((config, str(path)))

    print(f"{len(jobs)} runs to train with {args.workers} workers")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_run, config, path) for config, path in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            print(f"[{i}/{len(jobs)}] {future.result()}")

if __name__ == "__main__":
    main()
