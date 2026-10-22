from typing import Tuple

from tp4.types.float import Float

import numpy.typing as npt

# Standardization (z-score) per feature
def fit_z_score(inputs: npt.NDArray[Float]):
    """
    Compute the per-column mean and standard deviation of a (samples, features) matrix.

    Constant columns get a standard deviation of 1 so they map to 0 instead of dividing by zero.
    """
    mean = inputs.mean(axis=0)
    std = inputs.std(axis=0)
    std[std == 0] = 1.0
    return mean, std

def z_score(inputs: npt.NDArray[Float], mean: npt.NDArray[Float], std: npt.NDArray[Float]):
    """
    Standardize a (samples, features) matrix with statistics from fit_z_score.

    Test and validation sets must reuse the training set's statistics.
    """
    return (inputs - mean) / std

def z_score_normalization(inputs: npt.NDArray[Float]):
    """Fit a z-score on the samples and return (normalize, denormalize)."""
    mean, std = fit_z_score(inputs)
    return (lambda x: z_score(x, mean, std)), (lambda x: x * std + mean)
