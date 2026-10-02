import enum

from typing import List, Tuple, TypedDict
    
from tp3.types.activation_function import ActivationFunction
from tp3.types.derivative_function import DerivativeFunction

class PerceptronType(enum.Enum):
    SIMPLE = "simple_layer"
    MULTI_LAYER = "multi_layer"

class PerceptronDefinition(TypedDict):
    """
    Configuration for the perceptron model.

    Attributes:
        activation_functions: Activation and derivative functions to be used in the perceptron.
        learning_rate: The learning rate for the perceptron.
        epochs: Number of training epochs.
        datasets: Dataset paths for train, test, and validation sets along with the input and output columns.
    """
    type: PerceptronType
    activation_function: Tuple[ActivationFunction, DerivativeFunction] | None
    arquitecture: List[List[Tuple[ActivationFunction, DerivativeFunction]]] | None