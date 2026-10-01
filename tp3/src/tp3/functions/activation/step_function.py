from tp3.types.float import Float

import numpy as np

def step_function(x: Float) -> Float:
    """
    Compute the step activation function of x.

    Parameters:
    x : array-like
        Input value(s).

    Returns:
    array-like
        1 if x >= 0, 0 otherwise, applied to each element of x.
    """
    value = 1 if x >= 0 else 0
    return np.float64(value)

def step_function_derivative(x: Float) -> Float:
    """
    Compute the derivative of the step activation function of x.

    Parameters:
    x : array-like
        Input value(s).

    Returns:
    array-like
        0 for each element of x, since the derivative of the step function is not defined at x=0 and is 0 elsewhere.
    """
    return np.float64(1.0)
