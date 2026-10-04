from tp3.types.float import Float

import numpy as np
import numpy.typing as npt

def half_squared_error(arr1: npt.NDArray[Float], arr2: npt.NDArray[Float]) -> Float:
    """
    Compute the Halfed Squared Error (HSE) between two arrays.
    """
    return np.float64(0.5 * np.sum((arr1 - arr2) ** 2))

def half_squared_error_derivative(arr1: npt.NDArray[Float], arr2: npt.NDArray[Float]) -> npt.NDArray[Float]:
    return arr1 - arr2
