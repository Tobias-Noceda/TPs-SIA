from tp3.types.float import Float

import numpy as np
import numpy.typing as npt

def sigmoid_function(x: npt.NDArray[Float]) -> npt.NDArray[Float]:
    """
    Compute the Sigmoid function of x: 1 / (1 + exp(-x)).

    Parameters:
    x : Input value.

    Returns:
        Sigmoid applied to x.
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_function_derivative(x: npt.NDArray[Float]) -> npt.NDArray[Float]:
    """
    Compute the derivative of the Sigmoid function of x: sigmoid(x) * (1 - sigmoid(x)).

    Parameters:
    x : Input value.

    Returns:
        Derivative of the Sigmoid function applied to x.
    """
    return sigmoid_function(x) * (1 - sigmoid_function(x))
