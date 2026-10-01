import argparse

from tp3.config import load_config

import numpy as np

parser = argparse.ArgumentParser(description="Perceptron implementation.")

parser.add_argument("-c", "--config", dest="config", type=str, required=True, help="Configuration JSON file")

def main() -> None:
    args = parser.parse_args()

    data = load_config(args.config)


    functions = data["activation_functions"]
    activation_function, derivative_function = functions.functions
    
    learning_rate = data["learning_rate"]
    epochs = data["epochs"]
    dataset_path = data["dataset_path"]
