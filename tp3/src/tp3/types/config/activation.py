import enum

from typing import Tuple

from tp3.types.activation_function import ActivationFunction
from tp3.types.derivative_function import DerivativeFunction

from tp3.functions.activation.step_function import step_function, step_function_derivative
from tp3.functions.activation.linear_function import linear_function, linear_function_derivative
from tp3.functions.activation.non_linear.tanh_function import tanh_function, tanh_function_derivative
from tp3.functions.activation.non_linear.logistic_function import logistic_function, logistic_function_derivative

class ActivationFunctions(enum.Enum):
    STEP = "step"
    LINEAR = "linear"
    TANH = "tanh"
    LOGISTIC = "logistic"

    @property
    def functions(self):
        functions: dict[ActivationFunctions, Tuple[ActivationFunction, DerivativeFunction]] = {
            ActivationFunctions.STEP: (step_function, step_function_derivative),
            ActivationFunctions.LINEAR: (linear_function, linear_function_derivative),
            ActivationFunctions.TANH: (tanh_function, tanh_function_derivative),
            ActivationFunctions.LOGISTIC: (logistic_function, logistic_function_derivative),
        }
        
        return functions[self]