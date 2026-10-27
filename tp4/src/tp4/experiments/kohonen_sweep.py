import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from tp4 import create_model
from tp4.config import parse_config
from tp4.functions.metrics.hits import hits
from tp4.functions.metrics.u_matrix import u_matrix

import numpy as np

# Hyperparameter sweep of the Kohonen network: grid size × initial η × initial R × seeds.
# Each run is saved raw (results per epoch, hits, U-matrix, weights) so the analysis is done apart.
parser = argparse.ArgumentParser(description="Kohonen hyperparameter sweep.")

parser.add_argument("-c", "--config", dest="config", type=str, required=True, help="Base configuration JSON file")
parser.add_argument("-o", "--output", dest="output", type=str, default="res/output/kohonen_sweep", help="Output directory")
parser.add_argument("--grid-sizes", dest="grid_sizes", type=int, nargs="+", default=[4, 6, 8, 10])
parser.add_argument("--learning-rates", dest="learning_rates", type=float, nargs="+", default=[0.1, 0.5, 0.9], help="Initial η")
parser.add_argument("--radii", dest="radii", type=str, nargs="+", default=["1", "k/2", "k"], help="Initial R: a number, or 'k' / 'k/2' relative to the grid size")
parser.add_argument("--seeds", dest="seeds", type=int, default=3, help="Seeds per combination (0 … seeds - 1)")
parser.add_argument("--epochs", dest="epochs", type=int, default=None, help="Override the base config's epochs")
parser.add_argument("--workers", dest="workers", type=int, default=None, help="Parallel processes (default: all cores)")

def _radius(spec: str, k: int) -> float:
    if spec == "k":
        return float(k)
    if spec == "k/2":
        return k / 2
    return float(spec)

def _run(base: dict[str, Any], k: int, eta: float, radius: str, seed: int, output: Path) -> str:
    data = json.loads(json.dumps(base))
    data["kohonen"]["grid_size"] = k
    data["learning_rate"]["value"] = eta
    data["radius"]["value"] = _radius(radius, k)
    data["seed"] = seed

    # A decaying schedule must not end above where it starts
    for schedule in ("learning_rate", "radius"):
        final = data[schedule].get("hyperparameters", {}).get("final", None)
        if final is not None and final > data[schedule]["value"]:
            data[schedule]["hyperparameters"]["final"] = data[schedule]["value"]

    model, results, inputs, _, _, _ = create_model(*parse_config(data))
    winners = model.predict(inputs)

    name = f"k{k}_eta{eta:g}_R{radius.replace('/', 'over')}_s{seed}"
    np.savez(
        output / f"{name}.npz",
        grid_size=k,
        learning_rate=eta,
        radius=radius,
        initial_radius=data["radius"]["value"],
        seed=seed,
        results=np.array(results),
        hits=hits(winners, model.neurons),
        u_matrix=u_matrix(model.weights, model.grid_distances),
        weights=model.weights,
    )
    return name

def sweep(base: dict[str, Any], grid_sizes: list[int], learning_rates: list[float], radii: list[str], seeds: int, output: Path, workers: int | None = None) -> None:
    """Train every combination in parallel and save one .npz per run in output."""
    output.mkdir(parents=True, exist_ok=True)

    combinations = list(itertools.product(grid_sizes, learning_rates, radii, range(seeds)))
    print(f"Running {len(combinations)} trainings...")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run, base, k, eta, radius, seed, output) for k, eta, radius, seed in combinations]
        for future in futures:
            print(f"  done {future.result()}")

def main():
    args = parser.parse_args()

    with open(args.config, "r") as file:
        base = json.load(file)
    if args.epochs is not None:
        base["epochs"] = args.epochs

    sweep(base, args.grid_sizes, args.learning_rates, args.radii, args.seeds, Path(args.output) / "runs", args.workers)
