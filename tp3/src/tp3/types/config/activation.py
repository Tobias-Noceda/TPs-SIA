import enum

from typing import Tuple

from tp3.types.activation_function import ActivationFunction, DerivativeActivationFunction
from tp3.types.config.hyperparameters import Hyperparameters

from tp3.functions.activation.step_function import step_function, step_function_derivative
from tp3.functions.activation.linear_function import linear_function, linear_function_derivative
from tp3.functions.activation.non_linear.tanh_function import tanh_function, tanh_function_derivative
from tp3.functions.activation.non_linear.logistic_function import logistic_function, logistic_function_derivative
from tp3.functions.activation.non_linear.relu_function import relu_function, relu_function_derivative

import numpy as np

class ActivationFunctions(enum.Enum):
    STEP = "step"
    LINEAR = "linear"
    TANH = "tanh"
    LOGISTIC = "logistic"
    RELU = "relu"

    def functions(self, hyperparameters: Hyperparameters | None) -> Tuple[ActivationFunction, DerivativeActivationFunction]:
        beta = np.float64(hyperparameters.get("beta", 1.0)) if hyperparameters else np.float64(1.0)
        L = np.float64(hyperparameters.get("L", 1.0)) if hyperparameters else np.float64(1.0)

        functions: dict[ActivationFunctions, Tuple[ActivationFunction, DerivativeActivationFunction]] = {
            ActivationFunctions.STEP: (step_function, step_function_derivative),
            ActivationFunctions.LINEAR: (linear_function, linear_function_derivative),
            ActivationFunctions.TANH: (lambda x: tanh_function(x, beta), lambda x: tanh_function_derivative(x, beta)),
            ActivationFunctions.LOGISTIC: (lambda x: logistic_function(x, beta, L), lambda x: logistic_function_derivative(x, beta, L)),
            ActivationFunctions.RELU: (relu_function, relu_function_derivative)
        }
        
        return functions[self]
