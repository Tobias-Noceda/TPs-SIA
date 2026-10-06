"""
Ej. 1, parte A — capacidad de aprendizaje del perceptrón lineal vs. no lineal.

Entrena cada config × learning rate × semilla sobre TODO fraud_dataset.csv y guarda los
resultados crudos en `<out>/runs/*.npz`. No grafica nada: el análisis vive en
`tp3.analysis.ej1_learning`.
"""

import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tp3.config import load_config
from tp3.functions.normalization.z_score import fit_z_score, z_score
from tp3.models.neuron import Neuron
from tp3.models.simple_perceptron import SimplePerceptron
from tp3.parsers.dataset_parser import parse_dataset
from tp3.types.config.config import PerceptronConfig
from tp3.types.float import Float

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Ej1 parte A: aprendizaje del perceptrón lineal vs no lineal.")
parser.add_argument("-c", "--configs", nargs="+", required=True, help="Config JSON de cada perceptrón simple a comparar")
parser.add_argument("--lrs", nargs="+", type=float, default=None, help="Learning rates a barrer (pisa el del config)")
parser.add_argument("--seeds", type=int, default=5, help="Cantidad de semillas por combinación")
parser.add_argument("--epochs", type=int, default=None, help="Épocas (pisa el del config)")
parser.add_argument("--batch-size", type=int, default=None, help="Tamaño de batch (pisa el del config)")
parser.add_argument("-o", "--out", type=Path, default=Path("res/output/ej1_learning"), help="Directorio de salida")
parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count(), help="Procesos en paralelo")


def _load_data(config_path: str) -> tuple[PerceptronConfig, npt.NDArray[Float], npt.NDArray[Float], npt.NDArray[Float]]:
    """Config, inputs (z-scored if the config asks for it), BigModel target and flagged_fraud ground truth."""
    config = load_config(config_path)
    datasets = config["datasets"]
    columns = datasets["columns"]
    X, y = parse_dataset(datasets["train"], columns["input"], columns["output"])
    _, flagged = parse_dataset(datasets["train"], columns["input"], columns["expected"])

    if datasets.get("normalize", False):
        X = z_score(X, *fit_z_score(X))

    return config, X, y[:, 0], flagged[:, 0]


def _raw_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def _run(config_path: str, lr: float, seed: int, epochs: int | None, batch_size: int | None, out_dir: Path) -> Path:
    """Single training run. Rebuilds everything from the config path: activation lambdas don't pickle."""

    config, X, y, _ = _load_data(config_path)
    epochs = epochs if epochs is not None else config["epochs"]
    batch_size = batch_size if batch_size is not None else config["batch_size"]
    activation, derivative = config["perceptron"]["activation_function"]  # type: ignore[misc]
    loss_function, _ = config["loss_function"]

    np.random.seed(seed)  # weights and the per-epoch shuffle both draw from the global generator
    perceptron = SimplePerceptron(Neuron(activation, derivative), np.float64(lr), loss_function, X.shape[1], batch_size)

    start = time.perf_counter()
    try:
        history = perceptron.train(X, y, epochs)
        diverged = False
    except FloatingPointError:
        history, diverged = (), True
    elapsed = time.perf_counter() - start

    # Measured during each epoch, on each batch's output right before its update
    loss = np.array([e for _, e in history])
    h = X @ perceptron.weights + perceptron.bias
    predictions = perceptron.predict(X)

    raw = _raw_config(config_path)
    name = Path(config_path).stem
    meta = {
        "model": name,
        "activation": raw["perceptron"]["activation_function"]["name"],
        "learning_rate": lr,
        "seed": seed,
        "epochs": epochs,
        "epochs_run": len(loss),
        "batch_size": batch_size,
        "loss_function": raw["loss_function"],
        "train_seconds": elapsed,
        "diverged": diverged,
        "config": config_path,
    }

    out = out_dir / "runs" / f"{name}_lr{lr:g}_s{seed}.npz"
    np.savez_compressed(
        out,
        loss=loss,
        final_loss=loss_function(predictions, y) if not diverged else np.inf,
        predictions=predictions,
        h=h,
        weights=perceptron.weights,
        bias=np.array([perceptron.bias]),
        meta=json.dumps(meta),
    )
    return out


def _save_reference(config_path: str, out_dir: Path) -> None:
    """
    Dataset-level data shared by every run, saved once:
    - target / flagged: BigModel output and ground truth (flagged_fraud, only to color plots).
    - mean_baseline_{mse,loss}: error of always predicting mean(y), as MSE (= var(y), for R²) and
      with the config's loss function. A model near this learned nothing.
    - pc1 / pc1_scores: direction of max variance of the standardized features and each sample's
      projection on it, to plot the models' fitted curve against the real data in 1D.
    """
    config, X, y, flagged = _load_data(config_path)

    # PCA via SVD of the (already centered) feature matrix.
    # @link https://en.wikipedia.org/wiki/Principal_component_analysis#Singular_value_decomposition
    centered = X - X.mean(axis=0)
    _, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = Vt[0]
    explained = singular_values ** 2 / np.sum(singular_values ** 2)

    np.savez_compressed(
        out_dir / "reference.npz",
        target=y,
        flagged=flagged,
        pc1=pc1,
        pc1_scores=centered @ pc1,
        pc1_explained_variance=explained[0],
        feature_mean=X.mean(axis=0),
        mean_baseline_mse=np.var(y),
        mean_baseline_loss=config["loss_function"][0](np.full_like(y, y.mean()), y),
        loss_function=_raw_config(config_path)["loss_function"],
        input_columns=np.array(config["datasets"]["columns"]["input"]),
    )


def main() -> None:
    args = parser.parse_args()
    (args.out / "runs").mkdir(parents=True, exist_ok=True)

    jobs = []
    for config_path in args.configs:
        lrs = args.lrs if args.lrs is not None else [load_config(config_path)["learning_rate"]]
        jobs += [(config_path, lr, seed) for lr in lrs for seed in range(args.seeds)]

    _save_reference(args.configs[0], args.out)

    # One bar per worker would garble the output. tqdm reads TQDM_* when imported, so it has to be in the
    # environment the workers start with (fresh interpreters), not set inside them
    os.environ["TQDM_DISABLE"] = "1"
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=context) as pool:
        futures = [pool.submit(_run, c, lr, s, args.epochs, args.batch_size, args.out) for c, lr, s in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="runs"):
            future.result()

    print(f"{len(jobs)} corridas guardadas en {args.out / 'runs'}")


if __name__ == "__main__":
    main()
