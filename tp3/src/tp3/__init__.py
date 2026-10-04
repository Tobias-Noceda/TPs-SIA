import argparse

from tp3.config import load_config

from tp3.interfaces.perceptron import Perceptron
from tp3.models.neuron import Neuron
from tp3.models.simple_perceptron import SimplePerceptron
from tp3.models.multi_layer_perceptron import MultiLayerPerceptron

from tp3.parsers.dataset_parser import parse_dataset

import numpy as np
import matplotlib.pyplot as plt
from tp3.types.float import Float

parser = argparse.ArgumentParser(description="Perceptron implementation.")

parser.add_argument("-c", "--config", dest="config", type=str, required=True, help="Configuration JSON file")

def main() -> None:
    args = parser.parse_args()

    data = load_config(args.config)

    loss_functions = data["loss_function"]
    learning_rate = np.float64(data["learning_rate"])
    epochs = data["epochs"]
    datasets = data["datasets"]

    train_inputs, train_expected = parse_dataset(
        datasets["train"],
        input_columns=datasets["columns"]["input"],
        expected_output=datasets["columns"]["output"],
    )

    test_inputs, test_expected = parse_dataset(
        datasets["test"],
        input_columns=datasets["columns"]["input"],
        expected_output=datasets["columns"]["output"],
    )

    perceptron: Perceptron
    if data["perceptron"]["activation_function"] is not None:
        activation_function, derivative_function = data["perceptron"]["activation_function"]
        loss_function = loss_functions[0]

        perceptron = SimplePerceptron(activation_function, derivative_function, learning_rate, loss_function, len(datasets["columns"]["input"]))

        result = perceptron.train(
            tuple(train_inputs),
            tuple(e[0] for e in train_expected),
            epochs
        )
        print("Training complete. Final weights:", perceptron.weights)

        test_results = perceptron.test(
            tuple(test_inputs),
            tuple(e[0] for e in test_expected)
        )
    elif data["perceptron"]["arquitecture"] is not None:
        loss_function, derivative_loss_function = loss_functions
        arquitecture = data["perceptron"]["arquitecture"]

        layers = [
            tuple(
                Neuron(activation_function, derivative_function)
                for activation_function, derivative_function in layer
            )
            for layer in arquitecture
        ]

        perceptron = MultiLayerPerceptron(layers, learning_rate, loss_function, derivative_loss_function, len(datasets["columns"]["input"]))

        result = perceptron.train(
            tuple(train_inputs),
            tuple(train_expected),
            epochs
        )

        test_results = perceptron.test(
            tuple(np.array(input, dtype=np.float64) for input in test_inputs),
            tuple(np.array(e, dtype=np.float64) for e in test_expected)
        )
    else:
        raise ValueError("Perceptron configuration must have either an activation function or an architecture defined.")

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
        expected_output=datasets["columns"]["output"],
    )

    print(f"Perceptron accuracy: {test_results}")

    predictions = perceptron.predict(tuple(validation_inputs))

    print(f"Predicted: {[float(p) if isinstance(p, np.float64) else p for p in predictions]}")
