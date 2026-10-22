import argparse

from tp4.config import load_config

from tp4.functions.metrics.hits import hits
from tp4.functions.metrics.u_matrix import u_matrix
from tp4.models.kohonen import Kohonen
from tp4.parsers.dataset_parser import parse_dataset
from tp4.types.config.config import KohonenConfig
from tp4.visualization.kohonen import plot_components, plot_map, plot_training

import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Kohonen self-organizing map.")

parser.add_argument("-c", "--config", dest="config", type=str, required=True, help="Configuration JSON file")
parser.add_argument("-o", "--output", dest="output", type=str, required=False, help="Output .npz file for the trained map and results (optional)")
parser.add_argument("-p", "--plot", dest="plot", action="store_true", help="Show the map, U-matrix, training and component plots")

def create_model(data: KohonenConfig):
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
    model, results, inputs, labels, denormalize, seed = create_model(config)

    winners = model.predict(inputs)
    counts = hits(winners, model.neurons)
    distances = u_matrix(model.weights, model.grid_distances)

    final = results[-1]
    print(f"Quantization error: {final.quantization_error:.4f}")
    print(f"Topographic error: {final.topographic_error:.4f}")
    print(f"Dead neurons: {int((counts == 0).sum())} / {model.neurons}")

    # if labels is not None:
    #     print("Samples per neuron (row, column):")
    #     for neuron in np.flatnonzero(counts):
    #         row, column = model.coordinates(np.array([neuron]))[0]
    #         members = labels[winners == neuron]
    #         print(f"  ({row}, {column}) {len(members):>4}: {', '.join(members[:5])}{', ...' if len(members) > 5 else ''}")

    # # Weights back in the dataset's units when the normalization can be undone
    # weights = denormalize(model.weights) if denormalize is not None else model.weights

    # if args.output:
    #     np.savez(
    #         args.output,
    #         weights=model.weights,
    #         denormalized_weights=weights,
    #         positions=model.positions,
    #         winners=winners,
    #         coordinates=model.coordinates(winners),
    #         labels=labels if labels is not None else np.empty(0),
    #         columns=np.array(config["dataset"]["columns"]["input"]),
    #         hits=counts,
    #         u_matrix=distances,
    #         results=np.array(results),
    #         seed=str(seed.entropy),
    #     )

    # if args.plot:
    #     plot_training(results)
    #     plot_map(model.positions, model.topology, counts, distances)
    #     plot_components(model.positions, model.topology, weights, config["dataset"]["columns"]["input"])
    #     plt.show()
