from tp3.types.float import Float

import numpy as np

def linear_function(x: Float) -> Float:
    """
    Compute the linear (identity) activation function of x.

    Parameters:
    x : array-like
        Input value(s).

    Returns:
    array-like
        x unchanged.
    """
    return x

def linear_function_derivative(x: Float) -> Float:
    """
    Compute the derivative of the linear (identity) activation function of x.

    Parameters:
    x : array-like
        Input value(s).

    Returns:
    array-like
        1 for each element of x.
    """
    return np.float64(1.0)