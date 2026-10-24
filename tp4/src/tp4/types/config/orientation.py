import enum

from tp4.types.orientation_function import OrientationFunction

from tp4.functions.linear_algebra.orientation import keep_orientation, max_loading_positive

class Orientations(enum.Enum):
    MAX_LOADING_POSITIVE = "max_loading_positive"
    NONE = "none"

    def function(self) -> OrientationFunction:
        functions: dict[Orientations, OrientationFunction] = {
            Orientations.MAX_LOADING_POSITIVE: max_loading_positive,
            Orientations.NONE: keep_orientation,
        }

        return functions[self]
