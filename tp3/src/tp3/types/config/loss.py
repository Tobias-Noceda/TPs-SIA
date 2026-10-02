import enum

from typing import Tuple

from tp3.types.loss_function import LossFunction

from tp3.functions.loss.mean_squared_error import mean_squared_error

import numpy as np

class LossFunctions(enum.Enum):
    MSE = "mse"

    def function(self) -> LossFunction:
        functions: dict[LossFunctions, LossFunction] = {
            LossFunctions.MSE: mean_squared_error,
        }

        return functions[self]