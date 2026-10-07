import enum
from typing import Tuple

from tp3.functions.augmentation.shift import random_shift
from tp3.types.augmentation_function import AugmentationFunction
from tp3.types.config.aug_hyperparameters import AugmentationFunctionHyperparameters

class AugmentationFunctions(enum.Enum):
    NONE = "none"
    SHIFT = "shift"

    def function(self, shape: Tuple[int, ...], hyperparameters: AugmentationFunctionHyperparameters | None):
        hyper = hyperparameters or {}

        functions: dict[AugmentationFunctions, AugmentationFunction] = {
            AugmentationFunctions.NONE: lambda x, r: x,
            AugmentationFunctions.SHIFT: lambda x, r: random_shift(x, r, shape, hyper.get("max_shift", (1, 1))),
        }

        return functions[self]
