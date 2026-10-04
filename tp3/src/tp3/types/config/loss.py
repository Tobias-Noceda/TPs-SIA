import enum

from typing import Tuple

from tp3.types.loss_function import DerivativeLossFunction, LossFunction

from tp3.functions.loss.half_squared_error import half_squared_error, half_squared_error_derivative
from tp3.functions.loss.mean_squared_error import mean_squared_error, mean_squared_error_derivative

import numpy as np

class LossFunctions(enum.Enum):
    HSE = "hse"
    MSE = "mse"

    def functions(self) -> Tuple[LossFunction, DerivativeLossFunction]:
        functions: dict[LossFunctions, Tuple[LossFunction, DerivativeLossFunction]] = {
            LossFunctions.HSE: (half_squared_error, half_squared_error_derivative),
            LossFunctions.MSE: (mean_squared_error, mean_squared_error_derivative)
        }

        return functions[self]