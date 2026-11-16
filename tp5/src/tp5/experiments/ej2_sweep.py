import argparse
import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from tp5.config import parse_config
from tp5.persistence.run_file import save_run
from tp5.runners.basic import run_basic
from tp5.runners.variational import run_variational
from tp5.types.config.autoencoder import AutoencoderType

# Study of ex. 2: the same network trained as a plain autoencoder and as a variational one with
# several weights of the KL term (β), and with latent spaces of several sizes. It only trains and
# saves the models; analysis/ej2.py compares their latent spaces and what they generate.

type Variant = Callable[[Dict[str, Any]], None]

def beta(value: float) -> Variant:
    def apply(config: Dict[str, Any]) -> None:
        config["variational"] = {"beta": value}
    return apply

def plain() -> Variant:
    """Basic autoencoder with the same layers: the latent layer keeps only the k code neurons (no log Σ)."""
    def apply(config: Dict[str, Any]) -> None:
        config["autoencoder"]["type"] = "basic"
        config["autoencoder"]["encoder"]["arquitecture"][-1]["neurons"] //= 2
        config.pop("variational", None)
    return apply

def latent(size: int) -> Variant:
    """Latent space of `size` dimensions: the encoder outputs μ and log Σ, 2 · size neurons."""
    def apply(config: Dict[str, Any]) -> None:
        config["autoencoder"]["encoder"]["arquitecture"][-1]["neurons"] = 2 * size
    return apply

STUDIES: Dict[str, Dict[str, Variant]] = {
    # "ae" is the problem the VAE solves: no KL term, so nothing organizes the latent space
    "beta": {
        "ae": plain(),
        "0.1": beta(0.1),
        "1": beta(1),
        "5": beta(5),
        "25": beta(25),
    },
    "latent": {
        "2": latent(2),
        "8": latent(8),
        "32": latent(32),
    },
}

def _run(config: Dict[str, Any], path: str) -> str:
    # Workers run silently: one progress bar per process would interleave
    os.environ["TQDM_DISABLE"] = "1"
    try:
        parsed = parse_config(config)
        runner = run_variational if parsed["autoencoder"]["type"] == AutoencoderType.VARIATIONAL else run_basic
        run = runner(parsed)
        save_run(path, run)
        return f"done {Path(path).name} ({run.elapsed:.0f} s)"
    except FloatingPointError as error:
        return f"diverged {Path(path).name}: {error}"

def main() -> None:
    parser = argparse.ArgumentParser(description="Ex. 2: sweep of the variational autoencoder (β and latent size) against a plain autoencoder.")
    parser.add_argument("-c", "--config", default="res/config/ej2/emojis.json", help="Base config (variational, compact layers)")
    parser.add_argument("-o", "--output", default="res/output/ej2", help="Output directory (runs/ is created inside)")
    parser.add_argument("-s", "--seeds", type=int, default=1, help="Seeds per variant (0 … seeds - 1)")
    parser.add_argument("-e", "--epochs", type=int, default=None, help="Override the base config epochs")
    parser.add_argument("--studies", nargs="*", default=list(STUDIES), choices=list(STUDIES), help="Studies to run")
    parser.add_argument("-w", "--workers", type=int, default=os.cpu_count(), help="Parallel processes")
    parser.add_argument("--force", action="store_true", help="Re-run variants whose output already exists")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        base = json.load(file)
    if args.epochs is not None:
        base["epochs"] = args.epochs
    # Every variant trains the same number of epochs (their losses are not comparable)
    base["epsilon"] = 0.0
    # Validates the base before launching anything
    if parse_config(base, args.config)["autoencoder"]["type"] != AutoencoderType.VARIATIONAL:
        parser.error("the base config must be a variational autoencoder")

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
