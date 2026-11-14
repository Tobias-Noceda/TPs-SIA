import enum

from tp5.types.initialization_function import InitializationFunction

from tp5.functions.initialization.random import random_initialization
from tp5.functions.initialization.xavier import xavier_initialization

class InitializationFunctions(enum.Enum):
    RANDOM = "random"
    XAVIER = "xavier"

    def function(self) -> InitializationFunction:
        functions: dict[InitializationFunctions, InitializationFunction] = {
            InitializationFunctions.RANDOM: random_initialization,
            InitializationFunctions.XAVIER: xavier_initialization,
        }

        return functions[self]
