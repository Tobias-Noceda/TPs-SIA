from tp3.types.float import Float

import numpy as np
import numpy.typing as npt

def mean_squared_error(arr1: npt.NDArray[Float], arr2: npt.NDArray[Float]) -> Float:
    """
    Compute the Mean Squared Error (MSE) between two arrays.
    """
    return np.mean((arr1 - arr2) ** 2)

def mean_squared_error_derivative(arr1: npt.NDArray[Float], arr2: npt.NDArray[Float]) -> npt.NDArray[Float]:
    return 2 * (arr1 - arr2) / np.size(arr1)
