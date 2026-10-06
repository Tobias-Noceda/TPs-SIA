from typing import Tuple, TypedDict

from tp3.types.config.datasets import Datasets
from tp3.types.float import Float
from tp3.types.loss_function import DerivativeLossFunction, LossFunction
from tp3.types.learning_rate_function import LearningRateFactory
from tp3.types.config.perceptron import PerceptronDefinition
from tp3.types.classification_function import ClassificationFunction

class PerceptronConfig(TypedDict):
    """
    Configuration for the perceptron model.

    Attributes:
        activation_functions: Activation and derivative functions to be used in the perceptron.
        learning_rate: Factory of the learning rate function (one per parameter array).
        epochs: Number of training epochs.
        batch_size: Samples per weight update (1 = online, dataset size = batch).
        epsilon: Training stops once the epoch error is below it.
        seed: Seed for weight initialization and shuffling (None = random).
        datasets: Dataset paths for train, test, and validation sets along with the input and output columns.
    """
    perceptron: PerceptronDefinition
    loss_function: Tuple[LossFunction, DerivativeLossFunction]
    learning_rate: LearningRateFactory
    epochs: int
    batch_size: int
    epsilon: Float
    seed: int | None
    datasets: Datasets
    classification_function: ClassificationFunction
