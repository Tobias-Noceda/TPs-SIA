from typing import TypedDict

from tp4.types.float import Float
from tp4.types.initialization_function import InitializationFunction

class OjaDefinition(TypedDict):
    """
    Definition of the Oja neuron.

    Attributes:
        learning_rate: Constant η of every update.
        initialization: Draws the initial weights w.
    """
    learning_rate: Float
    initialization: InitializationFunction
