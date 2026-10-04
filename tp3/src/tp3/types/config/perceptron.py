import enum

from typing import List, Tuple, TypedDict
    
from tp3.types.activation_function import ActivationFunction, DerivativeActivationFunction

class PerceptronType(enum.Enum):
    SIMPLE = "single_layer"
    MULTI_LAYER = "multi_layer"

class PerceptronDefinition(TypedDict):
    """
    Definition of the perceptron model.

    Attributes:
        type: Kind of perceptron (single or multi layer).
        activation_function: Activation and derivative functions (single layer only).
        arquitecture: Activation and derivative functions per neuron, per layer (multi layer only).
    """
    type: PerceptronType
    activation_function: Tuple[ActivationFunction, DerivativeActivationFunction] | None
    arquitecture: List[List[Tuple[ActivationFunction, DerivativeActivationFunction]]] | None