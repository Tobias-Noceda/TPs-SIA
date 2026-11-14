from tp5.types.float import Float

import numpy as np
import numpy.typing as npt

def linear_function(x: npt.NDArray[Float]) -> npt.NDArray[Float]:
    """
    Compute the linear (identity) activation function of x.

    Parameters:
    x : Input value.

    Returns:
        x unchanged.
    """
    return x

def linear_function_derivative(x: npt.NDArray[Float]) -> npt.NDArray[Float]:
    """
    Compute the derivative of the linear (identity) activation function of x.

    Parameters:
    x : Input value.

    Returns:
        1.
    """
    return np.ones_like(x, dtype=np.float64)
