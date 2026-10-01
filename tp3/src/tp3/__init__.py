import argparse

from tp3.config import load_config

from tp3.models.simple_perceptron import SimplePerceptron
from tp3.functions.loss.loss_function import mean_squared_error

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Perceptron implementation.")

parser.add_argument("-c", "--config", dest="config", type=str, required=True, help="Configuration JSON file")

def main() -> None:
    args = parser.parse_args()

    data = load_config(args.config)


    functions = data["activation_functions"]
    activation_function, derivative_function = functions.functions
    
    learning_rate = np.float64(data["learning_rate"])
    epochs = data["epochs"]
    dataset_path = data["dataset_path"]

    perceptron = SimplePerceptron(mean_squared_error, learning_rate, activation_function, derivative_function)

    ## AND testing for step perceptron
    ZERO = np.float64(0)
    ONE = np.float64(1)

    inputs = (
        np.array([ZERO, ZERO], dtype=np.float64),
        np.array([ZERO, ONE], dtype=np.float64),
        np.array([ONE, ZERO], dtype=np.float64),
        np.array([ONE, ONE], dtype=np.float64)
    )
    expected = (ZERO, ZERO, ZERO, ONE)

    ## y = 2x testing for linear perceptron
    # numbers from 1 to 50, normalized to keep gradient steps stable
    # INPUT_SCALE = np.float64(50)
    # EXPECTED_SCALE = np.float64(100)

    # inputs = tuple(np.array([i], dtype=np.float64) / INPUT_SCALE for i in range(1, 51))
    # expected = tuple(np.float64(i * 2) / EXPECTED_SCALE for i in range(1, 51))

    ## y = tanh(x) testing for non-linear perceptron
    # INPUT_SCALE = np.float64(50)
    # EXPECTED_SCALE = np.float64(1)

    # inputs = tuple(np.array([i], dtype=np.float64) / INPUT_SCALE for i in range(1, 51))
    # expected = tuple(np.tanh(np.float64(i)) for i in range(1, 51))

    result = perceptron.train(inputs, expected, epochs)

    # Plot the loss over epochs
    epoch_numbers, errors = zip(*result)

    plt.plot(epoch_numbers, errors)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Perceptron training error")
    plt.show()

    print("Training complete. Final weights:", perceptron.weights)

    print("Testing the perceptron:")
    # predict_inputs = tuple(np.array([i], dtype=np.float64) / INPUT_SCALE for i in range(51, 61))
    # predictions = tuple(p * EXPECTED_SCALE for p in perceptron.predict(predict_inputs))
    predictions = perceptron.predict(inputs)
    print(f"Predicted: {predictions}")
