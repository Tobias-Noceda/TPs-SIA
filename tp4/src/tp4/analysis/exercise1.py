import argparse
import itertools
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, cast

from tp4.analysis import campaigns, eda
from tp4.analysis.kohonen_sweep import analyze
from tp4.config import load_config, parse_config
from tp4.experiments.kohonen_sweep import sweep
from tp4.functions.linear_algebra.cosine import cosine_similarity
from tp4.functions.linear_algebra.orientation import align
from tp4.functions.metrics.hits import hits
from tp4.functions.metrics.quantization_error import quantization_error
from tp4.functions.metrics.topographic_error import topographic_error
from tp4.functions.metrics.u_matrix import u_matrix
from tp4.models.kohonen import Kohonen
from tp4.models.oja import Oja
from tp4.models.pca import PCA
from tp4.parsers.dataset_parser import parse_dataset
from tp4.types.config.config import NeuralNetworkConfig, OjaConfig, PCAConfig
from tp4.types.samples import Samples
from tp4.visualization.exercise1 import plot_index_by_topic, plot_index_vs, plot_scores_by_topic, plot_strategy_comparison, plot_topic_shares
from tp4.visualization.kohonen import draw_grid, plot_categories, plot_components, plot_labels, plot_map, plot_projected_map, plot_training
from tp4.visualization.oja import plot_convergence, plot_index_comparison, plot_loadings
from tp4.visualization.pca import plot_biplot, plot_heatmap, plot_index, plot_scree

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy.random import MT19937, RandomState, SeedSequence

# Every figure, table and number of exercise 1 (creators), from the configs in res/config.
# Run from tp4/: figures and CSVs go to --output (versioned with the report), raw runs to --runs.
parser = argparse.ArgumentParser(description="Exercise 1: EDA, PCA, Kohonen, Oja and campaign decisions.")

parser.add_argument("-o", "--output", dest="output", type=str, default="docs/figures/ej1", help="Figures and tables")
parser.add_argument("--runs", dest="runs", type=str, default="res/output/ej1", help="Raw runs (sweep .npz)")
parser.add_argument("--pca", dest="pca", type=str, default="res/config/pca.json")
parser.add_argument("--kohonen", dest="kohonen", type=str, default="res/config/kohonen.json", help="Base config of the sweep")
parser.add_argument("--kohonen-final", dest="kohonen_final", type=str, default="res/config/kohonen_final.json", help="Chosen config")
parser.add_argument("--kohonen-reduced", dest="kohonen_reduced", type=str, default="res/config/kohonen_22.json", help="Reduced variable set (sensitivity)")
parser.add_argument("--oja", dest="oja", type=str, default="res/config/oja.json")
parser.add_argument("--skip-sweep", dest="skip_sweep", action="store_true", help="Reuse the sweep runs already in --runs")
parser.add_argument("--seeds", dest="seeds", type=int, default=10, help="Seeds of the final map for the stability analysis")

TOPIC_NAMES = list(campaigns.TOPICS)
TOPIC_COLUMNS = list(campaigns.TOPICS.values())

def _save(figure: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)

def _categories(frame: pd.DataFrame) -> npt.NDArray[np.intp]:
    """Index into TOPIC_NAMES of the dominant topic of every row."""
    return np.argmax(frame[TOPIC_COLUMNS].to_numpy(dtype=np.float64), axis=1)

def _read(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def _train_kohonen(data: dict[str, Any], raw: Samples) -> tuple[Kohonen, tuple, Any, Any]:
    """Same steps as tp4.create_model on the given raw samples: seed → normalize → train."""
    _, parsed = parse_config(data)
    config = cast(NeuralNetworkConfig, parsed)
    kohonen = config["kohonen"]
    normalize, denormalize = config["dataset"]["normalize"](raw)
    model = Kohonen(
        kohonen["grid_size"], kohonen["topology"], kohonen["similarity"], kohonen["neighborhood"],
        config["learning_rate"], config["radius"], kohonen["initialization"],
        RandomState(MT19937(SeedSequence(config["seed"]))),
    )
    results = model.train(normalize(raw), config["epochs"])
    return model, results, normalize, denormalize

def _override(data: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    data = json.loads(json.dumps(data))
    for key, value in overrides.items():
        if key == "grid_size":
            data["kohonen"]["grid_size"] = value
        elif key == "radius":
            data["radius"]["value"] = value
        else:
            data[key] = value
    return data

def _neighbor_pairs(winners: npt.NDArray[np.intp], grid_distances: npt.NDArray) -> npt.NDArray[np.bool_]:
    """(P, P) upper triangle: creators i < j mapped to the same or adjacent neurons."""
    close = grid_distances[winners[:, None], winners[None, :]] <= 1 + 1e-9
    return np.triu(close, k=1)

def _jaccard(a: npt.NDArray[np.bool_], b: npt.NDArray[np.bool_]) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))

def _purity(winners: npt.NDArray[np.intp], categories: npt.NDArray[np.intp]) -> float:
    """Share of creators whose dominant topic is the most common one in their neuron."""
    agree = 0
    for neuron in np.unique(winners):
        agree += np.bincount(categories[winners == neuron]).max()
    return agree / len(winners)

def _leave_one_out(data: dict[str, Any], raw: Samples, categories: npt.NDArray[np.intp], creator: int) -> tuple[bool, bool]:
    """
    A creator left out of the training and placed with the stored normalization (a new creator):
    (its neuron's weights share its dominant topic, the creators in that neuron or its neighbors mostly do).
    """
    mask = np.arange(len(raw)) != creator
    model, _, normalize, denormalize = _train_kohonen(data, raw[mask])
    neuron = int(model.predict(normalize(raw[creator:creator + 1]))[0])
    others = model.predict(normalize(raw[mask]))

    inputs = data["dataset"]["columns"]["input"]
    weights = pd.DataFrame(denormalize(model.weights[neuron:neuron + 1]), columns=inputs)
    near = categories[mask][model.grid_distances[neuron, others] <= 1 + 1e-9]
    majority = bool(len(near)) and int(np.bincount(near).argmax()) == categories[creator]
    return int(_categories(weights)[0]) == categories[creator], majority

# ── 1.1 EDA ─────────────────────────────────────────────────────────────────────

def section_eda(df: pd.DataFrame, inputs: list[str], output: Path) -> None:
    print("\n═══ 1.1 EDA ═══")
    directory = output / "eda"
    directory.mkdir(parents=True, exist_ok=True)
    eda.summary(df).round(4).to_csv(directory / "summary.csv")
    eda.checks(df, inputs, 0.9)
    eda.plots(df, inputs, directory)

    # Every numeric column (with the derived ones): shows why they're left out
    numeric = df.select_dtypes("number")
    _save(plot_heatmap(numeric.corr().to_numpy(), list(numeric.columns), list(numeric.columns), "Correlation matrix of every numeric column", annotate=False),
          directory / "correlation_all.png")

    categories = _categories(df)
    _save(plot_topic_shares(df[TOPIC_COLUMNS].to_numpy(), categories, TOPIC_NAMES, list(df["name"])), directory / "topics.png")

    # Followers say little about the audience: followers per dominant topic and vs Argentine share
    table = df.assign(topic=[TOPIC_NAMES[c] for c in categories]).groupby("topic")[
        ["followers", "average_reach", "argentina_audience_pct", "non_follower_reach_pct", "saves_per_1000", "content_tutorial_pct", "price_usd"]
    ].agg(["mean", "min", "max"]).round(1)
    table.to_csv(directory / "by_topic.csv")
    print(table.to_string())

    print("Correlation of followers with:")
    for column in ["average_reach", "price_usd", "argentina_audience_pct", "saves_per_1000", "profile_visits_per_1000", "content_tutorial_pct", "average_watch_time_seconds"]:
        print(f"  {np.corrcoef(df['followers'], df[column])[0, 1]:+.3f}  {column}")

# ── 1.1 PCA ─────────────────────────────────────────────────────────────────────

def section_pca(df: pd.DataFrame, path: str, output: Path) -> PCA:
    print("\n═══ 1.1 PCA ═══")
    _, parsed = load_config(path)
    config = cast(PCAConfig, parsed)
    assert config["pca"] is not None
    inputs = config["dataset"]["columns"]["input"]
    raw, labels = parse_dataset(config["dataset"]["path"], inputs, "name")
    assert labels is not None

    model = PCA(config["pca"]["components"], config["pca"]["orientation"])
    model.fit(raw)
    scores = model.scores(raw)
    ratio = model.explained_variance_ratio
    names = [f"PC{i + 1}" for i in range(len(ratio))]

    directory = output / "pca"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"eigenvalue": model.eigenvalues, "explained": ratio, "cumulative": np.cumsum(ratio)}, index=names).round(4).to_csv(directory / "eigenvalues.csv")
    pd.DataFrame(model.eigenvectors[:, :model.components], index=inputs, columns=names[:model.components]).round(4).to_csv(directory / "loadings.csv")
    pd.DataFrame(scores[:, :model.components], index=labels, columns=names[:model.components]).round(4).to_csv(directory / "scores.csv")

    print(f"Kept {model.components} components (Kaiser): {np.cumsum(ratio)[model.components - 1]:.1%} of the variance")
    for i in range(min(8, len(ratio))):
        print(f"  PC{i + 1}: λ = {model.eigenvalues[i]:.3f} ({ratio[i]:.1%}, cumulative {np.cumsum(ratio)[i]:.1%})")
    for component in range(3):
        loadings = model.eigenvectors[:, component]
        top = np.argsort(-np.abs(loadings))[:8]
        print(f"PC{component + 1}: " + ", ".join(f"{inputs[k]} {loadings[k]:+.2f}" for k in top))

    categories = _categories(df)
    by_topic = pd.DataFrame(scores[:, :3], columns=names[:3]).assign(topic=[TOPIC_NAMES[c] for c in categories]).groupby("topic").agg(["mean", "std"]).round(2)
    by_topic.to_csv(directory / "scores_by_topic.csv")
    print(by_topic.to_string())
    print(f"r(PC1, log followers) = {np.corrcoef(scores[:, 0], np.log(df['followers']))[0, 1]:+.3f}")

    _save(plot_scree(model.eigenvalues, model.components), directory / "scree.png")
    _save(plot_heatmap(model.eigenvectors[:, :model.components], inputs, names[:model.components], "Loadings of the kept components"), directory / "loadings.png")
    _save(plot_biplot(scores, model.eigenvectors, ratio, inputs, list(labels)), directory / "biplot.png")
    _save(plot_scores_by_topic(scores, ratio, categories, TOPIC_NAMES, list(labels), df["followers"].to_numpy(dtype=np.float64)), directory / "scores_by_topic.png")
    _save(plot_index(scores, list(labels)), directory / "pc1_index.png")
    _save(plot_index_by_topic(scores[:, 1], categories, TOPIC_NAMES, list(labels), "PC2"), directory / "pc2_by_topic.png")
    return model

# ── 1.2 Kohonen ─────────────────────────────────────────────────────────────────

def section_kohonen_sweep(args: argparse.Namespace, output: Path) -> None:
    print("\n═══ 1.2 Kohonen: hyperparameter sweep ═══")
    runs = Path(args.runs) / "kohonen_sweep"
    if not args.skip_sweep:
        if (runs / "runs").exists():
            shutil.rmtree(runs / "runs")
        sweep(_read(args.kohonen), [3, 4, 5, 6, 8], [0.1, 0.5, 0.9], ["1", "k/2", "k"], 3, runs / "runs")
    summary = analyze(runs)

    directory = output / "kohonen"
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(runs / "summary.csv", directory / "sweep_summary.csv")
    shutil.copy(runs / "plots" / "sweep.png", directory / "sweep.png")

    # Best combination per grid size (lowest QE + TE + dead share): the trade-off as k grows
    summary = summary.assign(rank=summary["quantization_error_mean"].rank() + summary["topographic_error_mean"].rank() + summary["dead_share_mean"].rank())
    best = summary.sort_values("rank").groupby("grid_size").head(1).sort_values("grid_size")
    columns = ["grid_size", "learning_rate", "radius", "quantization_error_mean", "quantization_error_std", "topographic_error_mean", "dead_neurons_mean", "dead_share_mean"]
    best[columns].round(4).to_csv(directory / "sweep_best_per_k.csv", index=False)
    print(best[columns].round(3).to_string(index=False))

def section_kohonen(df: pd.DataFrame, args: argparse.Namespace, pca: PCA, output: Path) -> tuple[Kohonen, npt.NDArray[np.intp], pd.DataFrame]:
    print("\n═══ 1.2 Kohonen: final map ═══")
    data = _read(args.kohonen_final)
    inputs = data["dataset"]["columns"]["input"]
    raw, labels = parse_dataset(data["dataset"]["path"], inputs, "name")
    assert labels is not None
    _, parsed = parse_config(data)
    topology = cast(NeuralNetworkConfig, parsed)["kohonen"]["topology"]

    model, results, normalize, denormalize = _train_kohonen(data, raw)
    winners = model.predict(normalize(raw))
    counts = hits(winners, model.neurons)
    distances = u_matrix(model.weights, model.grid_distances)
    weights = pd.DataFrame(denormalize(model.weights), columns=inputs)
    categories = _categories(df)
    neuron_topics = _categories(weights)

    print(f"k = {model.grid_size}: QE {results[-1].quantization_error:.4f}, TE {results[-1].topographic_error:.4f}, dead {int((counts == 0).sum())}/{model.neurons}")
    print(f"Topic purity of the neurons: {_purity(winners, categories):.1%}")
    agree = float(np.mean(neuron_topics[winners] == categories))
    print(f"Creators whose dominant topic equals their neuron's (weights) dominant topic: {agree:.1%}")

    directory = output / "kohonen"
    _save(plot_map(model.positions, topology, counts, distances), directory / "map.png")
    _save(plot_labels(model.positions, topology, distances, winners, list(labels), "U-matrix with the creators of each neuron"), directory / "u_matrix_names.png")
    _save(plot_categories(model.positions, topology, neuron_topics, TOPIC_NAMES, winners, list(labels), "Dominant topic of each neuron's weights"), directory / "topics.png")
    _save(plot_training(results), directory / "training.png")
    _save(plot_components(model.positions, topology, weights.to_numpy(), inputs), directory / "components.png")

    # Map vs PCA: PC1 / PC2 of each neuron's weights (in the original units, PCA standardizes them itself)
    projected = pca.scores(weights.to_numpy())
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    draw_grid(axes[0], model.positions, topology, projected[:, 0], "PC1 of each neuron's weights", annotate=False, label="PC1")
    draw_grid(axes[1], model.positions, topology, projected[:, 1], "PC2 of each neuron's weights", annotate=False, label="PC2")
    figure.tight_layout()
    _save(figure, directory / "pca_planes.png")
    _save(plot_projected_map(projected, pca.scores(raw), model.grid_distances, counts, categories, TOPIC_NAMES, pca.explained_variance_ratio), directory / "projected_map.png")

    # How well the grid axes follow PC1 / PC2: correlation of the neurons' PCs with their row / column
    rows, columns = model.positions[:, 1], model.positions[:, 0]
    for k, name in enumerate(["PC1", "PC2"]):
        print(f"  r({name}, column) = {np.corrcoef(projected[:, k], columns)[0, 1]:+.2f}, r({name}, row) = {np.corrcoef(projected[:, k], rows)[0, 1]:+.2f}")

    # Objective scores of every neuron (campaign view of the map)
    neuron_frame = weights.assign(estimated_argentina_reach=weights["average_reach"] * weights["argentina_audience_pct"] / 100)
    objectives = campaigns.objective_scores(neuron_frame, df)
    figure, axes = plt.subplots(1, len(objectives.columns), figsize=(4.6 * len(objectives.columns), 4.2))
    for ax, objective in zip(axes, objectives.columns):
        draw_grid(ax, model.positions, topology, objectives[objective].to_numpy(), objective, label="mean z-score")
    figure.tight_layout()
    _save(figure, directory / "objectives.png")

    neurons = pd.DataFrame({
        "row": model.positions[:, 1].astype(int), "column": model.positions[:, 0].astype(int),
        "hits": counts, "u_matrix": distances.round(3),
        "topic": [TOPIC_NAMES[c] for c in neuron_topics],
        "pc1": projected[:, 0].round(2), "pc2": projected[:, 1].round(2),
        **{objective: objectives[objective].round(2) for objective in objectives.columns},
        "creators": [", ".join(labels[winners == j]) for j in range(model.neurons)],
    })
    neurons.to_csv(directory / "neurons.csv", index=False)
    print(neurons[neurons["hits"] > 0].drop(columns=["creators"]).to_string(index=False))

    # Largest U-matrix borders: pairs of adjacent neurons farthest apart in the input space
    adjacent = np.argwhere(np.triu((model.grid_distances > 0) & (model.grid_distances <= 1 + 1e-9)))
    gaps = sorted(((float(np.linalg.norm(model.weights[i] - model.weights[j])), i, j) for i, j in adjacent), reverse=True)
    print("Largest distances between adjacent neurons:")
    for gap, i, j in gaps[:6]:
        print(f"  {gap:.2f}: ({i // model.grid_size}, {i % model.grid_size}) {TOPIC_NAMES[neuron_topics[i]]} ↔ ({j // model.grid_size}, {j % model.grid_size}) {TOPIC_NAMES[neuron_topics[j]]}")
    print("Smallest:")
    for gap, i, j in gaps[-3:]:
        print(f"  {gap:.2f}: ({i // model.grid_size}, {i % model.grid_size}) {TOPIC_NAMES[neuron_topics[i]]} ↔ ({j // model.grid_size}, {j % model.grid_size}) {TOPIC_NAMES[neuron_topics[j]]}")

    section_kohonen_stability(df, data, raw, model, winners, directory, args.seeds, args.kohonen_reduced)
    return model, winners, neurons

def section_kohonen_stability(df: pd.DataFrame, data: dict[str, Any], raw: Samples, reference: Kohonen, winners: npt.NDArray[np.intp], directory: Path, seeds: int, reduced_path: str) -> None:
    print("\n═══ 1.2 / 1.4 Kohonen: stability ═══")
    categories = _categories(df)
    base = _neighbor_pairs(winners, reference.grid_distances)

    rows = []
    pairs = []
    for seed in range(seeds):
        model, results, normalize, _ = _train_kohonen(_override(data, seed=seed), raw)
        found = model.predict(normalize(raw))
        pairs.append(_neighbor_pairs(found, model.grid_distances))
        rows.append({
            "variables": len(data["dataset"]["columns"]["input"]), "seed": seed,
            "quantization_error": float(results[-1].quantization_error), "topographic_error": float(results[-1].topographic_error),
            "dead": int((hits(found, model.neurons) == 0).sum()), "purity": _purity(found, categories),
            "jaccard_vs_final": _jaccard(pairs[-1], base),
        })
    agreement = [_jaccard(a, b) for a, b in itertools.combinations(pairs, 2)]

    # Reduced variable set with the same hyperparameters
    reduced = _override(_read(reduced_path), grid_size=data["kohonen"]["grid_size"], radius=data["radius"]["value"], seed=data["seed"])
    for key in ("learning_rate", "epochs"):
        reduced[key] = data[key]
    reduced_inputs = reduced["dataset"]["columns"]["input"]
    reduced_raw, _ = parse_dataset(reduced["dataset"]["path"], reduced_inputs)
    model, results, normalize, _ = _train_kohonen(reduced, reduced_raw)
    found = model.predict(normalize(reduced_raw))
    rows.append({
        "variables": len(reduced_inputs), "seed": reduced["seed"],
        "quantization_error": float(results[-1].quantization_error), "topographic_error": float(results[-1].topographic_error),
        "dead": int((hits(found, model.neurons) == 0).sum()), "purity": _purity(found, categories),
        "jaccard_vs_final": _jaccard(_neighbor_pairs(found, model.grid_distances), base),
    })

    table = pd.DataFrame(rows)
    table.round(4).to_csv(directory / "stability.csv", index=False)
    print(table.round(3).to_string(index=False))
    print(f"Neighbor pairs kept between seeds (Jaccard): mean {np.mean(agreement):.2f}, min {np.min(agreement):.2f}, max {np.max(agreement):.2f}")

    # A new creator placed on a map trained without it (leave-one-out, stored normalization)
    with ProcessPoolExecutor() as pool:
        placed = list(pool.map(_leave_one_out, itertools.repeat(data), itertools.repeat(raw), itertools.repeat(categories), range(len(raw))))
    neuron_topic, neighborhood = (np.array(column) for column in zip(*placed))
    print(f"Leave-one-out (new creator on a map trained without it): its neuron's weights share its dominant topic in {neuron_topic.mean():.0%}, "
          f"the creators around it mostly do in {neighborhood.mean():.0%}")
    pd.DataFrame({"creator": df["name"], "topic": [TOPIC_NAMES[c] for c in categories], "neuron_topic": neuron_topic, "neighborhood_topic": neighborhood}).to_csv(directory / "leave_one_out.csv", index=False)

# ── 1.3 Oja ─────────────────────────────────────────────────────────────────────

def section_oja(df: pd.DataFrame, path: str, pca: PCA, output: Path) -> npt.NDArray[np.float64]:
    print("\n═══ 1.3 Oja ═══")
    _, parsed = load_config(path)
    config = cast(OjaConfig, parsed)
    inputs = config["dataset"]["columns"]["input"]
    raw, labels = parse_dataset(config["dataset"]["path"], inputs, "name")
    assert labels is not None
    normalize, _ = config["dataset"]["normalize"](raw)
    samples = normalize(raw)
    component = pca.eigenvectors[:, 0]
    pc1 = pca.scores(raw)[:, 0]
    ranks = lambda v: np.argsort(np.argsort(v))

    directory = output / "oja"
    directory.mkdir(parents=True, exist_ok=True)

    # Learning rate: stability (η · ‖x‖² ≲ 1) and the bias of a constant η
    print(f"max ‖x‖² = {np.max(np.sum(samples ** 2, axis=1)):.1f}, 1 / (1.2 λ1) = {1 / (1.2 * pca.eigenvalues[0]):.4f}")
    rows = []
    for eta in [0.06, 0.02, 0.01, 0.005, 0.001, 0.0001]:
        model = Oja(eta, config["oja"]["initialization"], RandomState(MT19937(SeedSequence(config["seed"]))))
        try:
            results = model.train(samples, config["epochs"])
        except ValueError:
            rows.append({"learning_rate": eta, "diverges": True})
            continue
        weights = align(model.weights, component)
        history = np.array([r.weights for r in results])
        cosines = np.abs(history @ component) / np.linalg.norm(history, axis=1)
        index = samples @ weights
        rows.append({
            "learning_rate": eta, "diverges": False, "norm": float(np.linalg.norm(weights)),
            "cosine": float(cosine_similarity(weights, component)),
            "epochs_to_0.99": int(np.argmax(cosines >= 0.99)) + 1 if (cosines >= 0.99).any() else None,
            "pearson": float(np.corrcoef(index, pc1)[0, 1]), "spearman": float(np.corrcoef(ranks(index), ranks(pc1))[0, 1]),
        })
    table = pd.DataFrame(rows)
    table.to_csv(directory / "learning_rates.csv", index=False)
    print(table.to_string(index=False))

    model = Oja(config["oja"]["learning_rate"], config["oja"]["initialization"], RandomState(MT19937(SeedSequence(config["seed"]))))
    results = model.train(samples, config["epochs"])
    weights = align(model.weights, component)
    sign = np.sign(weights @ model.weights)
    index = model.predict(samples) * sign
    history = np.array([r.weights for r in results]) * sign
    cosines = np.abs(history @ component) / np.linalg.norm(history, axis=1)
    categories = _categories(df)

    print(f"η = {config['oja']['learning_rate']}: cos(w, PC1) = {cosine_similarity(weights, component):.5f}, ‖w‖ = {np.linalg.norm(weights):.4f}, "
          f"var(index) = {index.var():.3f} vs λ1 = {pca.eigenvalues[0]:.3f}")
    print(f"max |index − PC1| = {np.max(np.abs(index - pc1)):.3f}, Spearman(index, followers) = {np.corrcoef(ranks(index), ranks(df['followers']))[0, 1]:+.3f}")
    by_topic = pd.DataFrame({"index": index, "topic": [TOPIC_NAMES[c] for c in categories]}).groupby("topic")["index"].agg(["mean", "min", "max"]).round(2)
    print(by_topic.to_string())
    by_topic.to_csv(directory / "index_by_topic.csv")
    order = np.argsort(index)[::-1]
    pd.DataFrame({"creator": labels[order], "topic": [TOPIC_NAMES[c] for c in categories[order]], "oja": index[order].round(3), "pc1": pc1[order].round(3),
                  "followers": df["followers"].to_numpy()[order]}).to_csv(directory / "index.csv", index=False)
    print("Highest:", ", ".join(f"{labels[i]} ({TOPIC_NAMES[categories[i]]}, {index[i]:+.2f})" for i in order[:6]))
    print("Lowest: ", ", ".join(f"{labels[i]} ({TOPIC_NAMES[categories[i]]}, {index[i]:+.2f})" for i in order[::-1][:6]))

    _save(plot_convergence(cosines, np.array([r.norm for r in results])), directory / "convergence.png")
    _save(plot_loadings(weights, component, inputs), directory / "weights.png")
    _save(plot_index_by_topic(index, categories, TOPIC_NAMES, list(labels), "Oja index"), directory / "index.png")
    _save(plot_index_comparison(index, pc1, list(labels)), directory / "index_vs_pc1.png")
    _save(plot_index_vs(index, df["followers"].to_numpy(dtype=np.float64), categories, TOPIC_NAMES, list(labels), "Oja index", "followers", log=True), directory / "index_vs_followers.png")
    return index

# ── 1.4 Campaigns ───────────────────────────────────────────────────────────────

def section_campaigns(df: pd.DataFrame, model: Kohonen, winners: npt.NDArray[np.intp], neurons: pd.DataFrame, index: npt.NDArray[np.float64], output: Path) -> None:
    print("\n═══ 1.4 Campaigns ═══")
    directory = output / "campaigns"
    directory.mkdir(parents=True, exist_ok=True)
    count = 3
    scores = campaigns.objective_scores(df)
    topics = campaigns.dominant_topic(df)

    # Football: the three strategies of the statement, among football creators
    football = df[topics == "football"].assign(**{o: scores[o][topics == "football"].round(2) for o in scores.columns})
    football = football.assign(oja=index[topics.to_numpy() == "football"].round(2), neuron=[f"({w // model.grid_size}, {w % model.grid_size})" for w in winners[topics.to_numpy() == "football"]])
    columns = ["name", "neuron", "followers", "average_reach", "argentina_audience_pct", "non_follower_reach_pct", "shares_per_1000", "saves_per_1000",
               "content_tutorial_pct", "average_watch_time_seconds", "price_usd", "reach", "discovery", "depth", "conversion", "oja"]
    football[columns].sort_values("depth", ascending=False).to_csv(directory / "football_creators.csv", index=False)
    print(football[columns].sort_values("depth", ascending=False).to_string(index=False))
    print("Correlation between objective scores (football creators):")
    print(football[["reach", "discovery", "depth", "conversion", "oja"]].corr().round(2).to_string())

    selections = {
        "followers": campaigns.by_followers(df, count),
        "football by followers": df[topics == "football"].sort_values("followers", ascending=False).head(count),
        "reach": campaigns.select(df, "football", "reach", count),
        "discovery": campaigns.select(df, "football", "discovery", count),
        "depth": campaigns.select(df, "football", "depth", count),
    }
    # Niche creators are much cheaper: same budget as the massive selection
    selections["depth (same budget)"] = campaigns.select_budget(df, "football", "depth", float(selections["reach"]["price_usd"].sum()))
    comparison = campaigns.compare(df, selections, "football")
    comparison.to_csv(directory / "football_strategies.csv")
    print(comparison.to_string())
    metrics = {
        "on_topic": "Creators on football (share)", "argentina_audience_pct": "Argentine audience (%)", "targeted_reach": "Targeted reach (AR × football)",
        "non_follower_reach_pct": "Reach from non-followers (%)", "saves_per_1000": "Saves / 1000", "tutorial_pct": "Tutorial posts (%)",
        "cost_usd": "Cost of one post each (USD)", "usd_per_1000_targeted": "USD per 1000 targeted",
        "targeted_saves": "Saves from the targeted audience", "targeted_profile_visits": "Profile visits from the targeted audience",
    }
    _save(plot_strategy_comparison(comparison, metrics, f"Football campaign: {count} creators per strategy"), directory / "football_strategies.png")

    # Every client: followers-only vs profile-based selection (topic + its objective)
    goals = {"football": "depth", "makeup": "depth", "dogs": "conversion", "art": "depth"}
    rows = []
    for client, objective in goals.items():
        table = campaigns.compare(df, {"followers": campaigns.by_followers(df, count), objective: campaigns.select(df, client, objective, count), "reach": campaigns.select(df, client, "reach", count)}, client)
        rows.append(table.assign(client=client, strategy=table.index))
    clients = pd.concat(rows).set_index(["client", "strategy"])
    clients.to_csv(directory / "clients.csv")
    print(clients[["creators", "on_topic", "argentina_audience_pct", "targeted_reach", "cost_usd", "usd_per_1000_targeted", "targeted_profile_visits", "targeted_saves"]].to_string())

    # Dogs: shelters want enquiries and adoptions in Argentina
    dogs = df[topics == "dogs"].assign(conversion=scores["conversion"][topics == "dogs"].round(2), reach=scores["reach"][topics == "dogs"].round(2))
    dogs = dogs.sort_values("conversion", ascending=False)[["name", "followers", "argentina_audience_pct", "profile_visits_per_1000", "comments_per_1000", "saves_per_1000", "price_usd", "conversion", "reach"]]
    dogs.to_csv(directory / "dogs_creators.csv", index=False)
    print(dogs.to_string(index=False))

    # Top creators by followers: what they post and where their audience is
    top = campaigns.by_followers(df, 10).assign(topic=topics, oja=index[np.argsort(-df["followers"].to_numpy())[:10]].round(2))
    top[["name", "followers", "topic", "argentina_audience_pct", "non_follower_reach_pct", "saves_per_1000", "price_usd"]].to_csv(directory / "top_followers.csv", index=False)
    print(top[["name", "followers", "topic", "argentina_audience_pct", "saves_per_1000", "price_usd"]].to_string(index=False))

    # Alternatives on the map: neighbors of the best football depth creator
    best = int(np.flatnonzero(df["name"] == selections["depth"]["name"].iloc[0])[0])
    alternatives = campaigns.neighbors_on_map(winners, model.grid_distances, best)
    print(f"Map neighbors of {df['name'][best]}: " + ", ".join(f"{df['name'][i]} ({topics[i]})" for i in alternatives))

def main():
    args = parser.parse_args()
    plt.switch_backend("Agg")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    base = _read(args.kohonen_final)
    df = pd.read_csv(base["dataset"]["path"])
    inputs = base["dataset"]["columns"]["input"]

    section_eda(df, inputs, output)
    pca = section_pca(df, args.pca, output)
    section_kohonen_sweep(args, output)
    model, winners, neurons = section_kohonen(df, args, pca, output)
    index = section_oja(df, args.oja, pca, output)
    section_campaigns(df, model, winners, neurons, index, output)
    print(f"\nSaved figures and tables in {output}")
