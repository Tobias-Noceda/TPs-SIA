from typing import Tuple

from tp3.types.float import Float

import numpy as np
import numpy.typing as npt

# Standardization (z-score) per feature
def fit_z_score(inputs: npt.NDArray[Float]) -> Tuple[npt.NDArray[Float], npt.NDArray[Float]]:
    """
    Compute the per-column mean and standard deviation of a (samples, features) matrix.

    Constant columns get a standard deviation of 1 so they map to 0 instead of dividing by zero.
    """
    mean = inputs.mean(axis=0)
    std = inputs.std(axis=0)
    std[std == 0] = 1.0
    return mean, std

def z_score(inputs: npt.NDArray[Float], mean: npt.NDArray[Float], std: npt.NDArray[Float]) -> npt.NDArray[Float]:
    """
    Standardize a (samples, features) matrix with statistics from fit_z_score.

    Test and validation sets must reuse the training set's statistics.
    """
    return (inputs - mean) / std
