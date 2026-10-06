from tp3.types.float import Float

import numpy as np
import numpy.typing as npt

def tanh_function(x: npt.NDArray[Float], beta: Float = np.float64(1.0)) -> npt.NDArray[Float]:
    """
    Compute the hyperbolic tangent function of x: tanh(beta * x).

    Parameters:
    x : Input value.
    beta : float, optional
        The beta parameter for the hyperbolic tangent function.
        Default is 1.0.

    Returns:
        Hyperbolic tangent applied to x.
    """
    return np.tanh(beta * x)

def tanh_function_derivative(x: npt.NDArray[Float], beta: Float = np.float64(1.0)) -> npt.NDArray[Float]:
    """
    Compute the derivative of the hyperbolic tangent function of x: beta * (1 - tanh(beta * x)^2).

    Parameters:
    x : Input value.
    beta : float, optional
        The beta parameter for the hyperbolic tangent function.
        Default is 1.0.

    Returns:
        Derivative of the hyperbolic tangent function applied to x.
    """
    return beta * (1 - tanh_function(x, beta) ** 2)
