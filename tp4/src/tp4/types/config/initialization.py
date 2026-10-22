import enum

from tp4.types.initialization_function import InitializationFunction

from tp4.functions.initialization.samples import samples_initialization
from tp4.functions.initialization.uniform import uniform_initialization

class InitializationFunctions(enum.Enum):
    SAMPLES = "samples"
    UNIFORM = "uniform"

    def function(self) -> InitializationFunction:
        functions: dict[InitializationFunctions, InitializationFunction] = {
            InitializationFunctions.SAMPLES: samples_initialization,
            InitializationFunctions.UNIFORM: uniform_initialization,
        }

        return functions[self]
