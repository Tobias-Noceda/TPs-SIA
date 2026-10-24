import argparse
from pathlib import Path
from typing import cast

from tp4.config import load_config

from tp4.functions.activation.sign import sign
from tp4.functions.components.fixed import fixed_components
from tp4.functions.linear_algebra.cosine import cosine_similarity
from tp4.functions.linear_algebra.orientation import align
from tp4.functions.metrics.hits import hits
from tp4.functions.noise.flip import flip_noise
from tp4.models.hopfield import Hopfield
from tp4.models.kohonen import Kohonen
from tp4.models.oja import Oja
from tp4.parsers.dataset_parser import parse_dataset
from tp4.parsers.pattern_parser import parse_pattern
from tp4.models.pca import PCA
from tp4.types.config.config import HopfieldConfig, NeuralNetworkConfig, OjaConfig, PCAConfig
from tp4.types.config.method import Methods
from tp4.visualization.oja import plot_convergence, plot_index_comparison, plot_loadings
from tp4.visualization.pca import plot_biplot, plot_heatmap, plot_index, plot_scree

import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Unsupervised learning: Kohonen self-organizing map, PCA, Oja's rule and Hopfield network.")

parser.add_argument("-c", "--config", dest="config", type=str, required=True, help="Configuration JSON file")
parser.add_argument("-o", "--output", dest="output", type=str, required=False, help="Output .npz file for the trained model and results (optional)")
parser.add_argument("-p", "--plot", dest="plot", action="store_true", help="Show the plots of the method")

def create_model(method: Methods, data: NeuralNetworkConfig):
    if method is Methods.KOHONEN and "kohonen" in data:
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
    else:
        raise ValueError("Invalid method or missing configuration for the specified method.")

def main():
    args = parser.parse_args()
    method, config = load_config(args.config)

    if method is Methods.KOHONEN:
        run_kohonen(args, method, cast(NeuralNetworkConfig, config))
    elif method is Methods.PCA:
        run_pca(args, cast(PCAConfig, config))
    elif method is Methods.OJA:
        run_oja(args, cast(OjaConfig, config))
    elif method is Methods.HOPFIELD:
        run_hopfield(args, cast(HopfieldConfig, config))

def run_pca(args: argparse.Namespace, config: PCAConfig):
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

def run_oja(args: argparse.Namespace, config: OjaConfig):
    dataset = config["dataset"]
    variables = dataset["columns"]["input"]
    raw, labels = parse_dataset(dataset["path"], variables, dataset["columns"].get("label", None))
    normalize, _ = dataset["normalize"](raw)
    inputs = normalize(raw)

    model = Oja(config["oja"]["learning_rate"], config["oja"]["initialization"], RandomState(MT19937(SeedSequence(config["seed"]))))
    results = model.train(inputs, config["epochs"])

    # Reference: PC1 of the same samples (the PCA standardizes the raw inputs itself)
    pca = PCA(fixed_components(1))
    pca.fit(raw)
    component = pca.eigenvectors[:, 0]

    # The sign of w is arbitrary, like the eigenvector's: point it to the same side as PC1
    weights = align(model.weights, component)
    sign = np.sign(weights @ model.weights)
    index = model.predict(inputs) * sign
    pc1 = pca.scores(raw)[:, 0]

    cosine = cosine_similarity(weights, component)
    norm = np.linalg.norm(weights)
    print(f"‖w‖ = {norm:.6f}")
    print(f"cos(w, PC1) = {cosine:.6f} (angle {np.degrees(np.arccos(np.clip(cosine, -1, 1))):.3f}°)")
    print(f"Index variance {index.var():.4f} vs λ1 {pca.eigenvalues[0]:.4f} (equal when w = PC1, ‖w‖ = 1)")

    print("Weights (Oja vs PC1):")
    for k in np.argsort(-np.abs(component)):
        print(f"  {weights[k]:+.4f}  {component[k]:+.4f}  {variables[k]}")

    ranks = lambda values: np.argsort(np.argsort(values))
    print(f"Pearson r(index, PC1) = {np.corrcoef(index, pc1)[0, 1]:.6f}")
    print(f"Spearman ρ(index, PC1) = {np.corrcoef(ranks(index), ranks(pc1))[0, 1]:.6f}")

    if labels is not None:
        order = np.argsort(index)[::-1]
        print("Index highest:", ", ".join(f"{labels[i]} ({index[i]:+.2f})" for i in order[:5]))
        print("Index lowest: ", ", ".join(f"{labels[i]} ({index[i]:+.2f})" for i in order[::-1][:5]))

    history = np.array([result.weights for result in results]) * sign
    norms = np.array([result.norm for result in results])

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output,
            weights=weights,
            history=history,
            norms=norms,
            index=index,
            pc1=component,
            pc1_scores=pc1,
            labels=labels if labels is not None else np.empty(0),
            columns=np.array(variables),
        )

    if args.plot:
        cosines = np.abs(history @ component) / np.linalg.norm(history, axis=1)
        plot_convergence(cosines, norms)
        plot_loadings(weights, component, variables)
        plot_index(index, list(labels) if labels is not None else None, name="Oja index")
        plot_index_comparison(index, pc1, list(labels) if labels is not None else None)
        plt.show()

def run_hopfield(args: argparse.Namespace, config: HopfieldConfig):
    hopfield = config["hopfield"]

    parsed = [parse_pattern(path) for path in hopfield["patterns"]]
    names = [Path(path).stem for path in hopfield["patterns"]]
    shapes = {shape for _, shape in parsed}
    if len(shapes) != 1:
        raise ValueError(f"Every pattern must have the same shape, got {sorted(shapes)}.")
    if len(set(names)) != len(names):
        raise ValueError("Pattern file names must be unique: they name the patterns.")
    if hopfield["query"]["pattern"] not in names:
        raise ValueError(f"Query pattern '{hopfield['query']['pattern']}' is not stored: {names}.")
    
    shape = parsed[0][1]
    patterns = np.array([pattern for pattern, _ in parsed])

    p: int
    n: int
    p, n = patterns.shape

    # Single generator: same seed => same noisy queries and same asynchronous order
    random = RandomState(MT19937(SeedSequence(config["seed"])))
    model = Hopfield(hopfield["update"], hopfield["max_iterations"], random)
    model.store(patterns)

    print(f"N = {n} neurons ({shape[0]}x{shape[1]}), p = {p} patterns: p / N = {p / n:.4f} (capacity ≈ 0.15 → p ≤ {int(0.15 * n)})")

    # Orthogonality: overlap ξμ · ξν / N, 0 for orthogonal patterns (no crosstalk)
    overlap = patterns @ patterns.T / n
    off = np.abs(overlap[~np.eye(p, dtype=bool)])
    print("Overlap ξμ · ξν / N:")
    print("  " + " ".join(f"{name[:10]:>10}" for name in names))
    for name, row in zip(names, overlap):
        print(f"  {' '.join(f'{value:>+10.3f}' for value in row)}  {name}")
    if p > 1:
        print(f"|overlap| off the diagonal: max {off.max():.3f}, mean {off.mean():.3f}")

    # A stored pattern is a fixed point if one update leaves it unchanged
    print("Stability of the stored patterns (entries flipped by one synchronous update):")
    for name, pattern in zip(names, patterns):
        flipped = int(np.sum(sign(model.weights @ pattern, pattern) != pattern))
        print(f"  {name}: {'stable' if flipped == 0 else f'{flipped} entries flip'}")

    target = names.index(hopfield["query"]["pattern"])
    levels = hopfield["query"]["noise"]
    recalls = []
    results = []
    print(f"Queries from {names[target]}:")
    for level in levels:
        query = flip_noise(random, patterns[target], level)
        recall = model.recall(query)
        recalls.append(recall)

        final = recall.state @ patterns.T / n
        nearest = int(np.argmax(np.abs(final)))
        if final[nearest] == 1:
            result = names[nearest]
        elif final[nearest] == -1:
            result = f"inverse of {names[nearest]}"
        else:
            distance = int(np.sum(recall.state != np.sign(final[nearest]) * patterns[nearest]))
            result = f"spurious ({'' if final[nearest] > 0 else 'inverse of '}{names[nearest]} at {distance})"
        results.append(result)

        energies = " → ".join(f"{step.energy:.2f}" for step in recall.steps)
        print(f"  noise {level:>4.0%}: {len(recall.steps) - 1:>3} steps, {recall.convergence.value:<14} → {result}")
        print(f"    H: {energies}")

    overlaps = np.array([recall.state @ patterns.T / n for recall in recalls])
    # Recalls have different lengths: one (steps, N) states / (steps,) energies array per noise level
    states = np.empty(len(recalls), dtype=object)
    energies = np.empty(len(recalls), dtype=object)
    for k, recall in enumerate(recalls):
        states[k] = np.array([step.state for step in recall.steps])
        energies[k] = np.array([step.energy for step in recall.steps])

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output,
            weights=model.weights,
            patterns=patterns,
            names=np.array(names),
            shape=np.array(shape),
            overlap=overlap,
            levels=np.array(levels),
            queries=np.array([recall.steps[0].state for recall in recalls]),
            finals=np.array([recall.state for recall in recalls]),
            final_overlaps=overlaps,
            states=states,
            energies=energies,
            convergence=np.array([recall.convergence.value for recall in recalls]),
            results=np.array(results),
        )

def run_kohonen(args: argparse.Namespace, method: Methods, config: NeuralNetworkConfig):
    model, results, inputs, _, _, _ = create_model(method, config)

    winners = model.predict(inputs)
    counts = hits(winners, model.neurons)

    final = results[-1]
    print(f"Quantization error: {final.quantization_error:.4f}")
    print(f"Topographic error: {final.topographic_error:.4f}")
    print(f"Dead neurons: {int((counts == 0).sum())} / {model.neurons}")
