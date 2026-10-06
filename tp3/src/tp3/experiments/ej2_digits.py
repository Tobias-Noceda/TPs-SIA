"""
Ej. 2 — perceptrón multicapa para clasificar dígitos 0-9.

Lee la especificación de estudios (`res/config/ej2/studies.json`): una config base y, por estudio,
variantes que pisan algunos campos de la base. Corre cada variante × semilla y guarda los resultados
crudos en `<out>/runs/*.npz`; el análisis vive en `tp3.analysis.ej2_digits`.

También lo usa el ej. 3 (`res/config/ej3/studies.json`): `datasets.train` (o el `train` de una variante) puede ser
una lista de CSV, que se concatenan sin muestras repetidas, y cada variante puede pedir inicialización Xavier
(`weight_initialization`), desplazamientos aleatorios por época (`augment_shift`) y sobremuestreo de las clases
minoritarias (`oversample`).

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

from tp3.functions.augmentation.oversample import oversample
from tp3.functions.augmentation.shift import random_shift
from tp3.models.multi_layer_perceptron import MultiLayerPerceptron
from tp3.models.neuron import Neuron
from tp3.parsers.dataset_parser import parse_dataset
from tp3.types.config.activation import ActivationFunctions
from tp3.types.config.initialization import WeightInitializations
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


def train_files(spec: dict, config: dict) -> list[str]:
    """Training CSVs of a configuration: its own `train` or the spec's, always as a list."""
    files = config.get("train", spec["datasets"]["train"])
    return [files] if isinstance(files, str) else files


def data_path(spec: dict, config: dict, out: Path) -> Path:
    """Cache of a training set: data.npz for the spec's default one, data_<hash>.npz for the rest."""
    files = train_files(spec, config)
    if files == train_files(spec, {}):
        return out / "data.npz"
    return out / f"data_{hashlib.sha1(json.dumps(files).encode()).hexdigest()[:10]}.npz"


def load_data(spec: dict, configs: dict[str, dict], out: Path) -> dict[str, Path]:
    """
    Parses every CSV once (slow: one literal_eval per image) and caches, per distinct training set, its
    samples (several CSVs are concatenated dropping repeated images) plus the test set for every worker.
    Returns the cache of each configuration.
    """
    datasets = spec["datasets"]
    parsed: dict[str, tuple] = {}

    def parse(file: str) -> tuple:
        if file not in parsed:
            parsed[file] = parse_dataset(file, datasets["input"], datasets["label"], datasets["classes"])
        return parsed[file]

    paths = {}
    for cid, config in configs.items():
        path = paths[cid] = data_path(spec, config, out)
        if path.exists():
            continue
        X = np.vstack([parse(file)[0] for file in train_files(spec, config)])
        y = np.vstack([parse(file)[1] for file in train_files(spec, config)])
        # The CSVs share samples: keep the first occurrence of each image, in file order
        _, first = np.unique(X, axis=0, return_index=True)
        first.sort()
        X_test, y_test = parse(datasets["test"])
        np.savez(path, X=X[first], y=y[first], X_test=X_test, y_test=y_test)
    return paths


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
    initialization = WeightInitializations(config.get("weight_initialization", "uniform")).function()
    return MultiLayerPerceptron(
        layers, np.float64(config["learning_rate"]), loss_function, derivative, input_size, config["batch_size"], optimizer,
        initialization,
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
        # Each epoch sees a fresh resampling / shifted copy of the training set; the metrics below use the original
        samples = oversample(y_train) if config.get("oversample", False) else np.arange(len(X_train))
        X_epoch = random_shift(X_train[samples], config.get("augment_shift", 0))
        try:
            perceptron.train(X_epoch, y_train[samples], 1)
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

    (args.out / "runs").mkdir(exist_ok=True)
    configs, studies = resolve_studies(spec, args.only)
    print("Cargando datasets...")
    data_paths = load_data(spec, configs, args.out)
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

    _parallel([(_run, cid, configs[cid], seed, data_paths[cid], args.out) for cid, seed in jobs], args.jobs, "corridas")
    print(f"{len(jobs)} corridas ({len(configs)} configuraciones) guardadas en {args.out / 'runs'}")


if __name__ == "__main__":
    main()
