import enum

from tp3.functions.initialization.uniform import uniform_initialization
from tp3.functions.initialization.xavier import xavier_initialization
from tp3.types.weight_initialization import WeightInitialization

class WeightInitializations(enum.Enum):
    UNIFORM = "uniform"
    XAVIER = "xavier"

    def function(self) -> WeightInitialization:
        return {
            WeightInitializations.UNIFORM: uniform_initialization,
            WeightInitializations.XAVIER: xavier_initialization,
        }[self]
