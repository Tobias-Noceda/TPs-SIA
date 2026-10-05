from typing import Tuple, TypedDict

from tp3.types.config.datasets import Datasets
from tp3.types.float import Float
from tp3.types.loss_function import DerivativeLossFunction, LossFunction
from tp3.types.config.perceptron import PerceptronDefinition
from tp3.types.classification_function import ClassificationFunction

class PerceptronConfig(TypedDict):
    """
    Configuration for the perceptron model.

    Attributes:
        activation_functions: Activation and derivative functions to be used in the perceptron.
        learning_rate: The learning rate for the perceptron.
        epochs: Number of training epochs.
        datasets: Dataset paths for train, test, and validation sets along with the input and output columns.
    """
    perceptron: PerceptronDefinition
    loss_function: Tuple[LossFunction, DerivativeLossFunction]
    learning_rate: float
    epochs: int
    datasets: Datasets
    classification_function: ClassificationFunction
    epsilon: Float | None
    seed: int | None
