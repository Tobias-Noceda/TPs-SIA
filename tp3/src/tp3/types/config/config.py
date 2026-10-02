from typing import Callable, Tuple, TypedDict

from tp3.types.config.datasets import Datasets


class PerceptronConfig(TypedDict):
    """
    Configuration for the perceptron model.

    Attributes:
        activation_functions: Activation and derivative functions to be used in the perceptron.
        learning_rate: The learning rate for the perceptron.
        epochs: Number of training epochs.
        datasets: Dataset paths for train, test, and validation sets along with the input and output columns.
    """
    activation_functions: Tuple[Callable, Callable]
    learning_rate: float
    epochs: int
    datasets: Datasets
