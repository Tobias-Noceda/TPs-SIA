import enum

from tp4.types.neighborhood_function import NeighborhoodFunction

from tp4.functions.neighborhood.step import step_neighborhood
from tp4.functions.neighborhood.gaussian import gaussian_neighborhood

class NeighborhoodFunctions(enum.Enum):
    STEP = "step"
    GAUSSIAN = "gaussian"

    def function(self) -> NeighborhoodFunction:
        functions: dict[NeighborhoodFunctions, NeighborhoodFunction] = {
            NeighborhoodFunctions.STEP: step_neighborhood,
            NeighborhoodFunctions.GAUSSIAN: gaussian_neighborhood,
        }

        return functions[self]
