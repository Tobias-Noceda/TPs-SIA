import enum

from typing import Tuple

from tp5.types.loss_function import DerivativeLossFunction, LossFunction

from tp5.functions.loss.half_squared_error import half_squared_error, half_squared_error_derivative
from tp5.functions.loss.mean_squared_error import mean_squared_error, mean_squared_error_derivative
from tp5.functions.loss.binary_cross_entropy import binary_cross_entropy, binary_cross_entropy_derivative

class LossFunctions(enum.Enum):
    HSE = "hse"
    MSE = "mse"
    BCE = "bce"

    def functions(self) -> Tuple[LossFunction, DerivativeLossFunction]:
        functions: dict[LossFunctions, Tuple[LossFunction, DerivativeLossFunction]] = {
            LossFunctions.HSE: (half_squared_error, half_squared_error_derivative),
            LossFunctions.MSE: (mean_squared_error, mean_squared_error_derivative),
            LossFunctions.BCE: (binary_cross_entropy, binary_cross_entropy_derivative),
        }

        return functions[self]
