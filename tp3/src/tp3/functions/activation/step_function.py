from tp3.types.float import Float

import numpy as np

def step_function(x: Float) -> Float:
    """
    Compute the step activation function of x.

    Parameters:
    x : Input value.

    Returns:
        1 if x >= 0, 0 otherwise.
    """
    value = 1 if x >= 0 else 0
    return np.float64(value)

def step_function_derivative(x: Float) -> Float:
    """
    Compute the derivative of the step activation function of x.

    Parameters:
    x : Input value.

    Returns:
        1.
    """
    return np.float64(1.0)
