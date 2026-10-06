"""
Ej. 2 — perceptrón multicapa para clasificar dígitos 0-9.

Lee la especificación de estudios (`res/config/ej2/studies.json`): una config base y, por estudio,
variantes que pisan algunos campos de la base. Corre cada variante × semilla y guarda los resultados
crudos en `<out>/runs/*.npz`; el análisis vive en `tp3.analysis.ej2_digits`.

Evaluación: cada corrida entrena con TODO digits.csv (aunque no tenga 8s y tenga pocos 5s) y, una vez
entrenada, se evalúa en digits_test.csv ("producción"). Por época se guardan la pérdida y la accuracy de
train y la accuracy de test.
"""

import argparse
import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tp3.models.multi_layer_perceptron import MultiLayerPerceptron
from tp3.models.neuron import Neuron
from tp3.parsers.dataset_parser import parse_dataset
from tp3.types.config.activation import ActivationFunctions
from tp3.types.config.loss import LossFunctions
from tp3.types.config.optimizer import Optimizers
from tp3.types.float import Float

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Ej2: estudio de variantes del perceptrón multicapa en dígitos.")
parser.add_argument("-s", "--studies", type=Path, default=Path("res/config/ej2/studies.json"), help="Especificación de estudios")
parser.add_argument("--only", nargs="+", default=None, help="Correr solo estos estudios")
parser.add_argument("-o", "--out", type=Path, default=Path("res/output/ej2_digits"), help="Directorio de salida")
parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count(), help="Procesos en paralelo")


def load_data(spec: dict, out: Path) -> Path:
    """Parses both CSVs once (slow: one literal_eval per image) and caches them for every worker."""
    path = out / "data.npz"
    if path.exists():
        return path
    datasets = spec["datasets"]
    X, y = parse_dataset(datasets["train"], datasets["input"], datasets["label"], datasets["classes"])
    X_test, y_test = parse_dataset(datasets["test"], datasets["input"], datasets["label"], datasets["classes"])
    np.savez(path, X=X, y=y, X_test=X_test, y_test=y_test)
    return path


def config_id(config: dict) -> str:
    return hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]


def resolve_studies(spec: dict, only: list[str] | None) -> tuple[dict[str, dict], dict[str, dict]]:
    """
    Merges each variant over the base. Returns the unique configurations by id and, per study, its
    variants pointing to them (a variant equal to another study's one is trained only once).
    """
    configs: dict[str, dict] = {}
    studies: dict[str, dict] = {}
    for name, study in spec["studies"].items():
        if only is not None and name not in only:
            continue
        variants = []
        for variant in study["variants"]:
            config = spec["base"] | {k: v for k, v in variant.items() if k != "label"}
            configs[config_id(config)] = config
            variants.append({"label": variant["label"], "config_id": config_id(config)})
        studies[name] = {"title": study["title"], "seeds": study.get("seeds", spec["seeds"]), "variants": variants}
    return configs, studies


def build_perceptron(config: dict, input_size: int, classes: int) -> MultiLayerPerceptron:
    hidden = ActivationFunctions(config["hidden_activation"]).functions(None)
    output = ActivationFunctions(config["output_activation"]).functions(None)
    layers = [tuple(Neuron(*hidden) for _ in range(n)) for n in config["hidden"]]
    layers.append(tuple(Neuron(*output) for _ in range(classes)))

    loss_function, derivative = LossFunctions(config["loss_function"]).functions()
    optimizer = Optimizers(config["optimizer"]["name"]).factory(config["optimizer"].get("hyperparameters"))
    return MultiLayerPerceptron(
        layers, np.float64(config["learning_rate"]), loss_function, derivative, input_size, config["batch_size"], optimizer,
    )


def accuracy(outputs: npt.NDArray[Float], expected: npt.NDArray[Float]) -> float:
    """Fraction of samples whose most active output neuron is the labelled digit."""
    return float(np.mean(outputs.argmax(1) == expected.argmax(1)))


def _train(config: dict, seed: int, X_train, y_train, evaluation: dict[str, tuple]) -> dict[str, Any]:
    """Trains epoch by epoch, recording the training loss and the accuracy on each evaluation set."""
    np.random.seed(seed)  # weights and the per-epoch shuffle both draw from the global generator
    perceptron = build_perceptron(config, X_train.shape[1], y_train.shape[1])
    loss_function, _ = LossFunctions(config["loss_function"]).functions()

    epochs = config["epochs"]
    history = {"train_loss": np.full(epochs, np.nan), "learning_rate": np.full(epochs, np.nan)} | {f"{name}_accuracy": np.full(epochs, np.nan) for name in evaluation}
    diverged = False
    for epoch in range(epochs):
        try:
            perceptron.train(X_train, y_train, 1)
        except FloatingPointError:
            diverged = True
            break
        # Loss of the weights at the end of the epoch (train() reports the one seen while updating)
        history["train_loss"][epoch] = loss_function(perceptron.predict(X_train), y_train)
        # η used during the epoch (only the adaptive optimizer changes it)
        history["learning_rate"][epoch] = getattr(perceptron.layers[0].optimizer, "learning_rate", config["learning_rate"])
        for name, (X, y) in evaluation.items():
            history[f"{name}_accuracy"][epoch] = accuracy(perceptron.predict(X), y)

    predictions = {f"{name}_predictions": perceptron.predict(X).argmax(1) for name, (X, _) in evaluation.items()}
    return history | predictions | {"diverged": diverged}


def _run(config_id: str, config: dict, seed: int, data_path: Path, out: Path) -> None:
    data = np.load(data_path)
    X, y = data["X"], data["y"]
    result = _train(config, seed, X, y, {"train": (X, y), "test": (data["X_test"], data["y_test"])})
    meta = {"config_id": config_id, "config": config, "seed": seed, "diverged": result.pop("diverged")}
    np.savez_compressed(out / "runs" / f"{config_id}_s{seed}.npz", **result, test_labels=data["y_test"].argmax(1), meta=json.dumps(meta))


def _parallel(tasks: list[tuple], jobs: int, description: str) -> None:
    # One bar per worker would garble the output. tqdm reads TQDM_* when imported, so it has to be in the
    # environment the workers start with (fresh interpreters), not set inside them
    os.environ["TQDM_DISABLE"] = "1"
    # Parallelism comes from the process pool: multithreaded BLAS in every worker oversubscribes the cores
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as pool:
        futures = [pool.submit(*task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc=description):
            future.result()


def main() -> None:
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.studies) as f:
        spec = json.load(f)

    print("Cargando datasets...")
    data_path = load_data(spec, args.out)

    (args.out / "runs").mkdir(exist_ok=True)
    configs, studies = resolve_studies(spec, args.only)
    with open(args.out / "studies.json", "w") as f:
        json.dump({"spec": spec, "configs": configs, "studies": studies}, f, indent=2, ensure_ascii=False)

    # A configuration shared by several studies runs with the largest seed count among them
    seeds: dict[str, int] = {}
    for study in studies.values():
        for variant in study["variants"]:
            seeds[variant["config_id"]] = max(seeds.get(variant["config_id"], 0), study["seeds"])
    jobs = [(cid, seed) for cid, n in seeds.items() for seed in range(n)]
    # Biggest networks first so they don't end up alone at the tail
    jobs.sort(key=lambda job: -sum(configs[job[0]]["hidden"]))

    _parallel([(_run, cid, configs[cid], seed, data_path, args.out) for cid, seed in jobs], args.jobs, "corridas")
    print(f"{len(jobs)} corridas ({len(configs)} configuraciones) guardadas en {args.out / 'runs'}")


if __name__ == "__main__":
    main()
