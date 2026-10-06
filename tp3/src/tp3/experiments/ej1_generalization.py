"""
Ej. 1, parte B — generalización del perceptrón logístico con k-fold cross validation.

Por cada semilla parte fraud_dataset.csv en k folds estratificados por `flagged_fraud` y entrena
un perceptrón por fold. La normalización (z-score) se ajusta SOLO con el train de cada fold y se
reaplica a su test, para que el test no filtre información al entrenamiento. Guarda los resultados
crudos en `<out>/runs/*.npz`; el análisis vive en `tp3.analysis.ej1_generalization`.
"""

import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tp3.config import load_config
from tp3.functions.accuracy import accuracy, f1_score, precision, recall
from tp3.functions.classification.fraud import fraud_classification
from tp3.functions.loss.mean_squared_error import mean_squared_error
from tp3.functions.normalization.z_score import fit_z_score, z_score
from tp3.models.simple_perceptron import SimplePerceptron
from tp3.parsers.dataset_parser import parse_dataset
from tp3.types.float import Float

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Ej1 parte B: generalización con k-fold cross validation.")
parser.add_argument("-c", "--config", required=True, help="Config JSON del perceptrón elegido")
parser.add_argument("-k", "--folds", type=int, default=5, help="Cantidad de folds")
parser.add_argument("--seeds", type=int, default=3, help="Repeticiones del k-fold con particiones distintas")
parser.add_argument("--lr", type=float, default=None, help="Learning rate (pisa el del config)")
parser.add_argument("--epochs", type=int, default=None, help="Épocas (pisa el del config)")
parser.add_argument("--batch-size", type=int, default=None, help="Tamaño de batch (pisa el del config)")
parser.add_argument("-o", "--out", type=Path, default=Path("res/output/ej1_generalization"), help="Directorio de salida")
parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count(), help="Procesos en paralelo")


def _load_data(config_path: str) -> tuple[npt.NDArray[Float], npt.NDArray[Float], npt.NDArray[Float]]:
    """Raw inputs (NOT normalized: that happens per fold), BigModel target and flagged_fraud ground truth."""
    datasets = load_config(config_path)["datasets"]
    columns, path = datasets["columns"], datasets["train"]
    X, y = parse_dataset(path, columns["input"], columns["output"])
    _, flagged = parse_dataset(path, columns["input"], columns["expected"])
    return X, y[:, 0], flagged[:, 0]


def stratified_folds(labels: npt.NDArray[Float], k: int, seed: int) -> list[npt.NDArray[np.int64]]:
    """
    Test indices of each of the k folds. Each class is shuffled and split in k parts separately,
    so every fold keeps the dataset's fraud proportion (~12%): with plain k-fold a fold could get
    few frauds and its precision/recall would be noisy. Labels only drive the split, never training.
    """
    # @link https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation
    rng = np.random.default_rng(seed)
    folds: list[list[npt.NDArray[np.int64]]] = [[] for _ in range(k)]
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        for fold, part in zip(folds, np.array_split(indices, k)):
            fold.append(part)
    return [np.sort(np.concatenate(parts)) for parts in folds]


def _run(config_path: str, seed: int, fold: int, k: int, lr: float | None, epochs: int | None,
         batch_size: int | None, out_dir: Path) -> Path:
    """Trains on k-1 folds and evaluates on the remaining one. Rebuilt from the config path: activation lambdas don't pickle."""
    config = load_config(config_path)
    X, y, flagged = _load_data(config_path)
    lr = lr if lr is not None else config["learning_rate"]
    epochs = epochs if epochs is not None else config["epochs"]
    batch_size = batch_size if batch_size is not None else config["batch_size"]
    activation, derivative = config["perceptron"]["activation_function"]  # type: ignore[misc]
    loss_function, _ = config["loss_function"]

    test_idx = stratified_folds(flagged, k, seed)[fold]
    train_mask = np.ones(len(y), dtype=bool)
    train_mask[test_idx] = False

    # Statistics from the training folds only, then reused on the test fold
    mean, std = fit_z_score(X[train_mask])
    X_train, X_test = z_score(X[train_mask], mean, std), z_score(X[test_idx], mean, std)
    y_train, y_test = y[train_mask], y[test_idx]

    np.random.seed(seed * k + fold)  # weights and the per-epoch shuffle both draw from the global generator
    perceptron = SimplePerceptron(activation, derivative, np.float64(lr), loss_function, X.shape[1], batch_size)

    # One epoch per call so both sets can be evaluated with the weights at the end of every epoch.
    # MSE (a mean) instead of the config's HSE (a sum): train has k-1 times more samples than test.
    train_mse, test_mse = np.empty(epochs), np.empty(epochs)
    start = time.perf_counter()
    for epoch in range(epochs):
        perceptron.train(X_train, y_train, 1)
        train_mse[epoch] = mean_squared_error(perceptron.predict(X_train), y_train)
        test_mse[epoch] = mean_squared_error(perceptron.predict(X_test), y_test)
    elapsed = time.perf_counter() - start

    predictions = perceptron.predict(X_test)
    result = perceptron.test(X_test, flagged[test_idx], fraud_classification)

    meta = {
        "seed": seed,
        "fold": fold,
        "folds": k,
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_seconds": elapsed,
        "test_result": result,
        "metrics": {
            "accuracy": accuracy(result),
            "precision": precision(result),
            "recall": recall(result),
            "f1": f1_score(result),
        },
        "config": config_path,
    }

    out = out_dir / "runs" / f"s{seed}_f{fold}.npz"
    np.savez_compressed(
        out,
        train_mse=train_mse,
        test_mse=test_mse,
        test_loss=loss_function(predictions, y_test),
        test_idx=test_idx,
        predictions=predictions,
        weights=perceptron.weights,
        bias=np.array([perceptron.bias]),
        mean=mean,
        std=std,
        meta=json.dumps(meta),
    )
    return out


def main() -> None:
    args = parser.parse_args()
    (args.out / "runs").mkdir(parents=True, exist_ok=True)

    _, y, flagged = _load_data(args.config)
    np.savez_compressed(
        args.out / "reference.npz",
        target=y,
        flagged=flagged,
        input_columns=np.array(load_config(args.config)["datasets"]["columns"]["input"]),
    )

    jobs = [(seed, fold) for seed in range(args.seeds) for fold in range(args.folds)]

    # One bar per worker would garble the output. tqdm reads TQDM_* when imported, so it has to be in the
    # environment the workers start with (fresh interpreters), not set inside them
    os.environ["TQDM_DISABLE"] = "1"
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=context) as pool:
        futures = [
            pool.submit(_run, args.config, seed, fold, args.folds, args.lr, args.epochs, args.batch_size, args.out)
            for seed, fold in jobs
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="folds"):
            future.result()

    print(f"{len(jobs)} corridas ({args.seeds} semillas × {args.folds} folds) guardadas en {args.out / 'runs'}")


if __name__ == "__main__":
    main()
