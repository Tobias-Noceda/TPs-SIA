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
    return np.where(x >= 0, 1.0, 0.0)

def step_function_derivative(x: Float) -> Float:
    """
    Compute the derivative of the step activation function of x.

    Parameters:
    x : Input value.

    Returns:
        1.
    """
    return np.ones_like(x, dtype=np.float64)
