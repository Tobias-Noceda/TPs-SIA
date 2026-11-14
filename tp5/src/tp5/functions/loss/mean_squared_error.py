from tp5.types.float import Float

import numpy as np
import numpy.typing as npt

def mean_squared_error(arr1: npt.NDArray[Float], arr2: npt.NDArray[Float]):
    """
    Compute the Mean Squared Error (MSE) between two arrays.
    """
    return np.mean((arr1 - arr2) ** 2)

def mean_squared_error_derivative(arr1: npt.NDArray[Float], arr2: npt.NDArray[Float]):
    # Per-sample derivative: divided by the outputs count only, so a batch (samples, outputs)
    # yields one gradient per sample
    return 2 * (arr1 - arr2) / np.shape(arr1)[-1]
