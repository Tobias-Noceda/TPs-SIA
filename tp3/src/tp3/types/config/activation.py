import enum

from typing import Tuple

from tp3.types.activation_function import ActivationFunction
from tp3.types.config.hyperparameters import Hyperparameters
from tp3.types.derivative_function import DerivativeFunction

from tp3.functions.activation.step_function import step_function, step_function_derivative
from tp3.functions.activation.linear_function import linear_function, linear_function_derivative
from tp3.functions.activation.non_linear.tanh_function import tanh_function, tanh_function_derivative
from tp3.functions.activation.non_linear.logistic_function import logistic_function, logistic_function_derivative

import numpy as np

class ActivationFunctions(enum.Enum):
    STEP = "step"
    LINEAR = "linear"
    TANH = "tanh"
    LOGISTIC = "logistic"

    def functions(self, hyperparameters: Hyperparameters | None) -> Tuple[ActivationFunction, DerivativeFunction]:
        beta = np.float64(hyperparameters["beta"]) if hyperparameters and "beta" in hyperparameters else np.float64(1.0)
        L = np.float64(hyperparameters["L"]) if hyperparameters and "L" in hyperparameters else np.float64(1.0)

        functions: dict[ActivationFunctions, Tuple[ActivationFunction, DerivativeFunction]] = {
            ActivationFunctions.STEP: (step_function, step_function_derivative),
            ActivationFunctions.LINEAR: (linear_function, linear_function_derivative),
            ActivationFunctions.TANH: (lambda x: tanh_function(x, beta), lambda x: tanh_function_derivative(x, beta)),
            ActivationFunctions.LOGISTIC: (lambda x: logistic_function(x, beta, L), lambda x: logistic_function_derivative(x, beta, L)),
        }
        
        return functions[self]