import enum

from tp4.types.normalization_function import NormalizationFunction

from tp4.functions.normalization.identity import identity_normalization
from tp4.functions.normalization.z_score import z_score_normalization
from tp4.functions.normalization.min_max import min_max_normalization
from tp4.functions.normalization.unit_length import unit_length_normalization

class NormalizationFunctions(enum.Enum):
    NONE = "none"
    Z_SCORE = "z_score"
    MIN_MAX = "min_max"
    UNIT_LENGTH = "unit_length"

    def function(self) -> NormalizationFunction:
        functions: dict[NormalizationFunctions, NormalizationFunction] = {
            NormalizationFunctions.NONE: identity_normalization,
            NormalizationFunctions.Z_SCORE: z_score_normalization,
            NormalizationFunctions.MIN_MAX: min_max_normalization,
            NormalizationFunctions.UNIT_LENGTH: unit_length_normalization,
        }

        return functions[self]
