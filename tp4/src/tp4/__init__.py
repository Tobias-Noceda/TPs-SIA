import argparse
from pathlib import Path

from tp4.config import load_config

from tp4.functions.metrics.hits import hits
from tp4.models.kohonen import Kohonen
from tp4.parsers.dataset_parser import parse_dataset
from tp4.models.pca import PCA
from tp4.types.config.config import Config
from tp4.types.config.method import Methods
from tp4.visualization.pca import plot_biplot, plot_heatmap, plot_index, plot_scree

import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Unsupervised learning: Kohonen self-organizing map and PCA.")

parser.add_argument("-c", "--config", dest="config", type=str, required=True, help="Configuration JSON file")
parser.add_argument("-o", "--output", dest="output", type=str, required=False, help="Output .npz file for the trained model and results (optional)")
parser.add_argument("-p", "--plot", dest="plot", action="store_true", help="Show the plots of the method")

def create_model(data: Config):
    if "learning_rate" not in data or "radius" not in data or "epochs" not in data:
        raise ValueError("Kohonen configuration is missing learning_rate, radius or epochs in the config file.")

    if "kohonen" not in data:
        raise ValueError("Kohonen configuration is missing in the config file.")
    
    dataset = data["dataset"]
    kohonen = data["kohonen"]

    # Single generator for the whole model: same seed => same initial weights and same sample order
    seed = SeedSequence(data["seed"])
    random = RandomState(MT19937(seed))

    inputs, labels = parse_dataset(dataset["path"], dataset["columns"]["input"], dataset["columns"].get("label", None))
    normalize, denormalize = dataset["normalize"](inputs)
    inputs = normalize(inputs)

    model = Kohonen(
        kohonen["grid_size"],
        kohonen["topology"],
        kohonen["similarity"],
        kohonen["neighborhood"],
        data["learning_rate"],
        data["radius"],
        kohonen["initialization"],
        random,
    )

    results = model.train(inputs, data["epochs"])

    return model, results, inputs, labels, denormalize, seed

def main():
    args = parser.parse_args()
    config = load_config(args.config)

    if config["method"] is Methods.KOHONEN:
        run_kohonen(args, config)
    elif config["method"] is Methods.PCA:
        run_pca(args, config)

def run_pca(args: argparse.Namespace, config: Config):
    if "pca" not in config or config["pca"] is None:
        raise ValueError("PCA configuration is missing in the config file.")
    
    dataset = config["dataset"]
    variables = dataset["columns"]["input"]
    inputs, labels = parse_dataset(dataset["path"], variables, dataset["columns"].get("label", None))

    model = PCA(config["pca"]["components"], config["pca"]["orientation"])
    model.fit(inputs)
    scores = model.scores(inputs)
    ratio = model.explained_variance_ratio

    print("Component  eigenvalue  explained  cumulative")
    for i, (eigenvalue, share, cumulative) in enumerate(zip(model.eigenvalues, ratio, np.cumsum(ratio))):
        mark = "*" if i < model.components else " "
        print(f"{mark} PC{i + 1:<6} {eigenvalue:>10.4f} {share:>10.2%} {cumulative:>11.2%}")
    print(f"Kept components: {model.components} (*)")

    for component in range(min(2, len(variables))):
        loadings = model.eigenvectors[:, component]
        print(f"PC{component + 1} loadings:")
        for k in np.argsort(-np.abs(loadings)):
            print(f"  {loadings[k]:+.3f}  {variables[k]}")

    if labels is not None:
        order = np.argsort(scores[:, 0])[::-1]
        print("PC1 highest:", ", ".join(f"{labels[i]} ({scores[i, 0]:+.2f})" for i in order[:5]))
        print("PC1 lowest: ", ", ".join(f"{labels[i]} ({scores[i, 0]:+.2f})" for i in order[::-1][:5]))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output,
            eigenvalues=model.eigenvalues,
            eigenvectors=model.eigenvectors,
            explained_variance_ratio=ratio,
            components=model.components,
            correlation=model.correlation,
            scores=scores,
            mean=model.mean,
            std=model.std,
            labels=labels if labels is not None else np.empty(0),
            columns=np.array(variables),
        )

    if args.plot and labels is not None:
        plot_scree(model.eigenvalues, model.components)
        names = [f"PC{i + 1}" for i in range(model.components)]
        plot_heatmap(model.eigenvectors[:, :model.components], variables, names, "Loadings of the kept components")
        if len(variables) >= 2:
            plot_biplot(scores, model.eigenvectors, ratio, variables, list(labels))
        plot_index(scores, list(labels))
        plt.show()

def run_kohonen(args: argparse.Namespace, config: Config):
    model, results, inputs, _, _, _ = create_model(config)

    winners = model.predict(inputs)
    counts = hits(winners, model.neurons)

    final = results[-1]
    print(f"Quantization error: {final.quantization_error:.4f}")
    print(f"Topographic error: {final.topographic_error:.4f}")
    print(f"Dead neurons: {int((counts == 0).sum())} / {model.neurons}")
