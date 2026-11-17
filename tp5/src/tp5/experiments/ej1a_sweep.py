import argparse
import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from tp5.config import parse_config
from tp5.experiments.parallel import prepare_workers
from tp5.persistence.run_file import save_run
from tp5.runners.basic import run_basic

# One-factor-at-a-time study of ex. 1a.2: every variant changes a single aspect of the base config,
# so each family of runs isolates the effect of that aspect (architecture, optimizer, loss, ...).

type Variant = Callable[[Dict[str, Any]], None]

def _layer(neurons: int, activation: str) -> Dict[str, Any]:
    return {"neurons": neurons, "activation_function": {"name": activation}}

def _activations(config: Dict[str, Any]) -> Tuple[str, str, str]:
    """(hidden, latent, output) activation names of a base config written in the compact layer form."""
    encoder = config["autoencoder"]["encoder"]["arquitecture"]
    decoder = config["autoencoder"]["decoder"]["arquitecture"]
    if not all(isinstance(layer, dict) for layer in encoder + decoder):
        raise ValueError("The sweep needs the base config layers in the compact form { neurons, activation_function }.")
    hidden = encoder[0]["activation_function"]["name"] if len(encoder) > 1 else "tanh"
    return hidden, encoder[-1]["activation_function"]["name"], decoder[-1]["activation_function"]["name"]

def _hidden_sizes(config: Dict[str, Any]) -> List[int]:
    return [layer["neurons"] for layer in config["autoencoder"]["encoder"]["arquitecture"][:-1]]

def _set_network(config: Dict[str, Any], hidden: List[int], hidden_activation: str, latent_activation: str, output_activation: str) -> None:
    """Encoder input → hidden → 2 (latent), decoder 2 → reversed hidden → pixels (mirror)."""
    pixels = config["autoencoder"]["decoder"]["arquitecture"][-1]["neurons"]
    config["autoencoder"]["encoder"]["arquitecture"] = [_layer(n, hidden_activation) for n in hidden] + [_layer(2, latent_activation)]
    config["autoencoder"]["decoder"]["arquitecture"] = [_layer(n, hidden_activation) for n in reversed(hidden)] + [_layer(pixels, output_activation)]

def architecture(hidden: List[int]) -> Variant:
    def apply(config: Dict[str, Any]) -> None:
        _set_network(config, hidden, *_activations(config))
    return apply

def activation(position: str, name: str) -> Variant:
    def apply(config: Dict[str, Any]) -> None:
        hidden_activation, latent_activation, output_activation = _activations(config)
        names = {"hidden": hidden_activation, "latent": latent_activation, "output": output_activation}
        names[position] = name
        _set_network(config, _hidden_sizes(config), names["hidden"], names["latent"], names["output"])
    return apply

def key(name: str, value: Any) -> Variant:
    def apply(config: Dict[str, Any]) -> None:
        config[name] = copy.deepcopy(value)
    return apply

STUDIES: Dict[str, Dict[str, Variant]] = {
    "architecture": {
        "35-2-35": architecture([]),
        "35-10-2": architecture([10]),
        "35-20-2": architecture([20]),
        "35-25-15-2": architecture([25, 15]),
        "35-30-20-10-2": architecture([30, 20, 10]),
    },
    "optimizer": {
        "gd_0.5": key("learning_rate", {"type": "constant", "value": 0.5}),
        "momentum_0.1": key("learning_rate", {"type": "momentum", "value": 0.1, "hyperparameters": {"alpha": 0.9}}),
        "adam_0.001": key("learning_rate", {"type": "adam", "value": 0.001}),
        "adam_0.01": key("learning_rate", {"type": "adam", "value": 0.01}),
    },
    "loss": {
        "mse": key("loss_function", "mse"),
        "bce": key("loss_function", "bce"),
    },
    "batch_size": {
        "1": key("batch_size", 1),
        "8": key("batch_size", 8),
        "32": key("batch_size", 32),
    },
    "latent_activation": {
        "linear": activation("latent", "linear"),
        "tanh": activation("latent", "tanh"),
        "logistic": activation("latent", "logistic"),
    },
    "hidden_activation": {
        "tanh": activation("hidden", "tanh"),
        "relu": activation("hidden", "relu"),
        "logistic": activation("hidden", "logistic"),
    },
    "initialization": {
        "random": key("initialization", "random"),
        "xavier": key("initialization", "xavier"),
    },
}

def _run(config: Dict[str, Any], path: str) -> str:
    try:
        save_run(path, run_basic(parse_config(config)))
        return f"done {Path(path).name}"
    except FloatingPointError as error:
        return f"diverged {Path(path).name}: {error}"

def main() -> None:
    parser = argparse.ArgumentParser(description="Ex. 1a.2: one-factor-at-a-time sweep of the basic autoencoder.")
    parser.add_argument("-c", "--config", default="res/config/ej1a/autoencoder.json", help="Base config (compact layers)")
    parser.add_argument("-o", "--output", default="res/output/ej1a", help="Output directory (runs/ is created inside)")
    parser.add_argument("-s", "--seeds", type=int, default=5, help="Seeds per variant (0 … seeds - 1)")
    parser.add_argument("-e", "--epochs", type=int, default=None, help="Override the base config epochs")
    parser.add_argument("--studies", nargs="*", default=list(STUDIES), choices=list(STUDIES), help="Studies to run")
    parser.add_argument("-w", "--workers", type=int, default=os.cpu_count(), help="Parallel processes")
    parser.add_argument("--force", action="store_true", help="Re-run variants whose output already exists")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        base = json.load(file)
    if args.epochs is not None:
        base["epochs"] = args.epochs
    # Validates the base (and the compact layer form) before launching anything
    parse_config(base, args.config)
    _activations(base)

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

    prepare_workers(min(args.workers, len(jobs)))
    print(f"{len(jobs)} runs to train with {args.workers} workers")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_run, config, path) for config, path in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            print(f"[{i}/{len(jobs)}] {future.result()}")
