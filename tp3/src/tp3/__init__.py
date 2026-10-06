import argparse
import json

from tp3.config import load_config

from tp3.functions.accuracy import accuracy
from tp3.interfaces.perceptron import Perceptron
from tp3.models.neuron import Neuron
from tp3.models.simple_perceptron import SimplePerceptron
from tp3.models.multi_layer_perceptron import MultiLayerPerceptron
from tp3.types.config.config import PerceptronConfig
from tp3.types.config.perceptron import PerceptronType

from tp3.parsers.dataset_parser import parse_dataset
from tp3.functions.normalization.z_score import fit_z_score, z_score

import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Perceptron implementation.")

parser.add_argument("-c", "--config", dest="config", type=str, required=True, help="Configuration JSON file")
parser.add_argument("-o", "--output", dest="output", type=str, required=False, help="Output file for results (optional)")

def create_model(data: PerceptronConfig):
    loss_functions = data["loss_function"]
    learning_rate = data["learning_rate"]
    epochs = data["epochs"]
    batch_size = data["batch_size"]
    classification_function = data["classification_function"]
    datasets = data["datasets"]
    epsilon = data["epsilon"]
    test_in_train = data["test_in_train"]

    # Single generator for the whole model: same seed => same weights and same shuffling
    seed = SeedSequence(data["seed"])
    random = RandomState(MT19937(seed))

    perceptron_type = data["perceptron"]["type"]

    # Sorry :/
    arquitecture = list()
    if perceptron_type is PerceptronType.MULTI_LAYER:
        arquitecture = data["perceptron"]["arquitecture"]

    if arquitecture is None:
        raise ValueError("A multi_layer perceptron requires an 'arquitecture'.")

    labels_count = 1 if perceptron_type is PerceptronType.SIMPLE else len(arquitecture[-1])

    print("Parsing datasets, this might take a while...")

    train_inputs, train_expected = parse_dataset(
        datasets["train"],
        input_columns=datasets["columns"]["input"],
        expected_output=datasets["columns"]["output"],
        labels_count=labels_count
    )

    test_inputs, test_expected = parse_dataset(
        datasets["test"],
        input_columns=datasets["columns"]["input"],
        expected_output=datasets["columns"]["expected"],
        labels_count=labels_count
    )

    # Statistics come from the training set only, then are reused for test and validation
    normalize = datasets.get("normalize", False)
    mean = np.empty(0)
    std = np.empty(0)
    if normalize:
        mean, std = fit_z_score(train_inputs)
        train_inputs = z_score(train_inputs, mean, std)
        test_inputs = z_score(test_inputs, mean, std)

    test_function = None

    perceptron: Perceptron
    if perceptron_type is PerceptronType.SIMPLE:
        activation = data["perceptron"]["activation_function"]
        if activation is None:
            raise ValueError("A single_layer perceptron requires an 'activation_function'.")
        
        activation_function, derivative_function = activation
        loss_function = loss_functions[0]

        perceptron = SimplePerceptron(
            Neuron(activation_function, derivative_function),
            learning_rate,
            loss_function,
            train_inputs.shape[1],
            batch_size,
            epsilon,
            random
        )

        if test_in_train:
            test_function = lambda: accuracy(perceptron.test(test_inputs, test_expected[:, 0], classification_function))
        
        result = perceptron.train(train_inputs, train_expected[:, 0], epochs, classification_function if test_in_train else None, test_function)
        print("Training complete. Final weights:", perceptron.weights)

        test_results = perceptron.test(test_inputs, test_expected[:, 0], classification_function)
    elif perceptron_type is PerceptronType.MULTI_LAYER:
        loss_function, derivative_loss_function = loss_functions
        layers = [
            tuple(
                Neuron(activation_function, derivative_function)
                for activation_function, derivative_function in layer
            )
            for layer in arquitecture
        ]

        perceptron = MultiLayerPerceptron(layers, learning_rate, loss_function, derivative_loss_function, train_inputs.shape[1], batch_size, epsilon, random)

        if test_in_train:
            test_function = lambda: accuracy(perceptron.test(test_inputs, test_expected, classification_function))

        result = perceptron.train(train_inputs, train_expected, epochs, classification_function if test_in_train else None, test_function)

        test_results = perceptron.test(test_inputs, test_expected, classification_function)
    else:
        raise ValueError(f"Unknown perceptron type: {perceptron_type}")

    return perceptron, labels_count, result, test_results, datasets, normalize, mean, std, seed

def main():
    args = parser.parse_args()
    perceptron, labels_count, result, test_results, datasets, normalize, mean, std, seed = create_model(load_config(args.config))

    # Plot the loss over epochs
    print("Plotting training error over epochs...")

    epoch_numbers, errors, acc, test_acc = zip(*result)

    plt.plot(epoch_numbers, errors)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Perceptron training error")
    plt.show()

    validation_inputs, _ = parse_dataset(
        datasets["validation"],
        input_columns=datasets["columns"]["input"],
        expected_output=datasets["columns"]["expected"],
        labels_count=labels_count
    )

    if normalize:
        validation_inputs = z_score(validation_inputs, mean, std)

    if args.output:
        with open(args.output, "w") as f:
            output = {
                "training_results": result,
                "test_results": test_results,
                "acuracy": accuracy(test_results),
                "seed": seed.entropy,
            }
            f.write(json.dumps(output))
            

    print(f"Perceptron accuracy: {test_results}")

    predictions = perceptron.predict(validation_inputs)

    print(f"Predicted: {predictions.tolist()}")
