import enum

from tp3.types.Initialization_function import InitializationFunction

from tp3.functions.initailization.random import random_initialization
from tp3.functions.initailization.xavier import xavier_initialization

class InitializationFunctions(enum.Enum):
    RANDOM = "random"
    XAVIER = "xavier"

    def function(self) -> InitializationFunction:
        functions: dict[InitializationFunctions, InitializationFunction] = {
            InitializationFunctions.RANDOM: random_initialization,
            InitializationFunctions.XAVIER: xavier_initialization,
        }

        return functions[self]
