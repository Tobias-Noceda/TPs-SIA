from tp4.types.float import Float

import numpy as np
import numpy.typing as npt

# Min-max feature scaling to [0, 1] per feature
# @link https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
def min_max_normalization(inputs: npt.NDArray[Float]):
    """
    Fit a min-max scaling on the samples and return (normalize, denormalize).

    Constant columns get a range of 1 so they map to 0 instead of dividing by zero.
    """
    low = inputs.min(axis=0)
    span = inputs.max(axis=0) - low
    span = np.where(span == 0, 1.0, span)
    return (lambda x: (x - low) / span), (lambda x: x * span + low)
