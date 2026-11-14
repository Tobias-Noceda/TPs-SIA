import enum

from functools import cache
from typing import Tuple

from tp5.types.activation_function import ActivationFunction, DerivativeActivationFunction
from tp5.types.config.hyperparameters import Hyperparameters

from tp5.functions.activation.linear_function import linear_function, linear_function_derivative
from tp5.functions.activation.non_linear.tanh_function import tanh_function, tanh_function_derivative
from tp5.functions.activation.non_linear.logistic_function import logistic_function, logistic_function_derivative
from tp5.functions.activation.non_linear.relu_function import relu_function, relu_function_derivative

import numpy as np

class ActivationFunctions(enum.Enum):
    LINEAR = "linear"
    TANH = "tanh"
    LOGISTIC = "logistic"
    RELU = "relu"

    def functions(self, hyperparameters: Hyperparameters | None) -> Tuple[ActivationFunction, DerivativeActivationFunction]:
        beta = float(hyperparameters.get("beta", 1.0)) if hyperparameters else 1.0
        L = float(hyperparameters.get("L", 1.0)) if hyperparameters else 1.0

        return _functions(self, beta, L)

# Cached so neurons configured alike share the same function objects: Layer groups neurons by
# function identity to apply each activation once over the whole batch
@cache
def _functions(name: ActivationFunctions, beta: float, L: float) -> Tuple[ActivationFunction, DerivativeActivationFunction]:
    b = np.float64(beta)
    l = np.float64(L)

    functions: dict[ActivationFunctions, Tuple[ActivationFunction, DerivativeActivationFunction]] = {
        ActivationFunctions.LINEAR: (linear_function, linear_function_derivative),
        ActivationFunctions.TANH: (lambda x: tanh_function(x, b), lambda x: tanh_function_derivative(x, b)),
        ActivationFunctions.LOGISTIC: (lambda x: logistic_function(x, b, l), lambda x: logistic_function_derivative(x, b, l)),
        ActivationFunctions.RELU: (relu_function, relu_function_derivative),
    }

    return functions[name]
