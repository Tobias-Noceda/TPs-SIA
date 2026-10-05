import argparse

from tp3.config import load_config

from tp3.interfaces.perceptron import Perceptron
from tp3.models.neuron import Neuron
from tp3.models.simple_perceptron import SimplePerceptron
from tp3.models.multi_layer_perceptron import MultiLayerPerceptron

from tp3.parsers.dataset_parser import parse_dataset

import numpy as np
import matplotlib.pyplot as plt
from tp3.types.config.perceptron import PerceptronType
from tp3.types.float import Float

parser = argparse.ArgumentParser(description="Perceptron implementation.")

parser.add_argument("-c", "--config", dest="config", type=str, required=True, help="Configuration JSON file")

def main() -> None:
    args = parser.parse_args()

    data = load_config(args.config)

    loss_functions = data["loss_function"]
    learning_rate = np.float64(data["learning_rate"])
    epochs = data["epochs"]
    classification_function = data["classification_function"]
    datasets = data["datasets"]

    perceptron_type = data["perceptron"]["type"]

    # Sorry :/
    arquitecture = list()
    if perceptron_type is PerceptronType.MULTI_LAYER:
        arquitecture = data["perceptron"]["arquitecture"]

    if arquitecture is None:
        raise ValueError("A multi_layer perceptron requires an 'arquitecture'.")

    labels_count = 1 if perceptron_type is PerceptronType.SIMPLE else len(arquitecture[-1])

    train_inputs, train_expected = parse_dataset(
        datasets["train"],
        input_columns=datasets["columns"]["input"],
        expected_output=datasets["columns"]["output"],
        labels_count=labels_count,
    )

    test_inputs, test_expected = parse_dataset(
        datasets["test"],
        input_columns=datasets["columns"]["input"],
        expected_output=datasets["columns"]["expected"],
        labels_count=labels_count,
    )

    perceptron: Perceptron
    if perceptron_type is PerceptronType.SIMPLE:
        activation = data["perceptron"]["activation_function"]
        if activation is None:
            raise ValueError("A single_layer perceptron requires an 'activation_function'.")
        
        activation_function, derivative_function = activation
        loss_function = loss_functions[0]

        perceptron = SimplePerceptron(activation_function, derivative_function, learning_rate, loss_function, len(train_inputs[0]))

        result = perceptron.train(
            tuple(train_inputs),
            tuple(e[0] for e in train_expected),
            epochs
        )
        print("Training complete. Final weights:", perceptron.weights)

        test_results = perceptron.test(
            tuple(test_inputs),
            tuple(e[0] for e in test_expected),
            classification_function
        )
    elif perceptron_type is PerceptronType.MULTI_LAYER:
        loss_function, derivative_loss_function = loss_functions
        layers = [
            tuple(
                Neuron(activation_function, derivative_function)
                for activation_function, derivative_function in layer
            )
            for layer in arquitecture
        ]

        perceptron = MultiLayerPerceptron(layers, learning_rate, loss_function, derivative_loss_function, len(train_inputs[0]))

        result = perceptron.train(
            tuple(train_inputs),
            tuple(train_expected),
            epochs
        )

        test_results = perceptron.test(
            tuple(np.array(input, dtype=np.float64) for input in test_inputs),
            tuple(np.array(e, dtype=np.float64) for e in test_expected),
            classification_function
        )
    else:
        raise ValueError(f"Unknown perceptron type: {perceptron_type}")

    # Plot the loss over epochs
    print("Plotting training error over epochs...")

    epoch_numbers, errors = zip(*result)

    plt.plot(epoch_numbers, errors)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Perceptron training error")
    plt.show()

    validation_inputs, _ = parse_dataset(
        datasets["validation"],
        input_columns=datasets["columns"]["input"],
        expected_output=datasets["columns"]["expected"],
        labels_count=labels_count,
    )

    print(f"Perceptron accuracy: {test_results}")

    predictions = perceptron.predict(tuple(validation_inputs))

    print(f"Predicted: {[float(p) if isinstance(p, np.float64) else p for p in predictions]}")
