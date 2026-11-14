from tp5.types.float import Float

import numpy as np
import numpy.typing as npt

def relu_function(x: npt.NDArray[Float]) -> npt.NDArray[Float]:
    """
    Compute the ReLU function of x: max(0, x).

    Parameters:
    x : Input value.

    Returns:
        ReLU applied to x.
    """
    return np.maximum(0, x)

def relu_function_derivative(x: npt.NDArray[Float]) -> npt.NDArray[Float]:
    """
    Compute the derivative of the ReLU function of x: 1 if x > 0 else 0.

    Parameters:
    x : Input value.

    Returns:
        Derivative of the ReLU function applied to x.
    """
    return (x > 0).astype(float)
