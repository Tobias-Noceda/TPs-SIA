from typing import TypedDict

from tp3.types.config.activation import ActivationFunctions
from tp3.types.config.datasets import DatasetPaths


class PerceptronConfig(TypedDict):
    """
    Configuration for the perceptron model.

    Attributes:
        activation_functions: Activation and derivative functions to be used in the perceptron.
        learning_rate: The learning rate for the perceptron.
        epochs: Number of training epochs.
        dataset_path: Dataset paths for train, test, and validation sets.
    """
    activation_functions: ActivationFunctions
    learning_rate: float
    epochs: int
    dataset_path: DatasetPaths
