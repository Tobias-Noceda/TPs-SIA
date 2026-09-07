import enum

from tp2.interfaces.algorithms.breeding import Breeding

from tp2.algorithms.breeding.single_point import single_point_crossover
from tp2.algorithms.breeding.double_point import double_point_crossover
from tp2.algorithms.breeding.uniform import uniform_crossover
from tp2.algorithms.breeding.anular import anular_crossover

class BreedingAlgorithms(enum.Enum):
    SINGLE_POINT = "single_point"
    DOUBLE_POINT = "double_point"
    UNIFORM = "uniform"
    ANULAR = "anular"

    @property
    def function(self):

        functions: dict[BreedingAlgorithms, Breeding] = {
            BreedingAlgorithms.SINGLE_POINT: single_point_crossover,
            BreedingAlgorithms.DOUBLE_POINT: double_point_crossover,
            BreedingAlgorithms.UNIFORM: uniform_crossover,
            BreedingAlgorithms.ANULAR: anular_crossover,
        }
        
        return functions[self]