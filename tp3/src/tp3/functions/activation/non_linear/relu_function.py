from tp3.types.float import Float

import numpy as np

def relu_function(x: Float) -> Float:
    """
    Compute the ReLU function of x: max(0, x).

    Parameters:
    x : Input value.

    Returns:
        ReLU applied to x.
    """
    return np.maximum(0, x)

def relu_function_derivative(x: Float) -> Float:
    """
    Compute the derivative of the ReLU function of x: 1 if x > 0 else 0.

    Parameters:
    x : Input value.

    Returns:
        Derivative of the ReLU function applied to x.
    """
    return (x > 0).astype(float)
