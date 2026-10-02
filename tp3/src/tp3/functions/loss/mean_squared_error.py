from tp3.types.float import Float

import numpy as np
import numpy.typing as npt

def mean_squared_error(arr1: npt.NDArray[Float], arr2: npt.NDArray[Float]) -> Float:
    """
    Compute the Mean Squared Error (MSE) between two arrays.
    """
    return np.float64(0.5 * np.sum((arr1 - arr2) ** 2))
