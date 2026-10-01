from tp3.types.float import Float

import numpy as np

def tanh_function(x: Float) -> Float:
    """
    Compute the hyperbolic tangent function of x, with beta=1: tanh(x).

    Parameters:
    x : array-like
        Input value(s).

    Returns:
    array-like
        Hyperbolic tangent applied to each element of x.
    """
    return np.tanh(1 * x) # Beta = 1 explisitly

def tanh_function_derivative(x: Float) -> Float:
    """
    Compute the derivative of the hyperbolic tangent function of x, with beta=1: 1 - tanh(x)^2.

    Parameters:
    x : array-like
        Input value(s).

    Returns:
    array-like
        Derivative of the hyperbolic tangent function applied to each element of x.
    """
    return 1 - tanh_function(x) ** 2
