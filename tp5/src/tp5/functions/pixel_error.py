from tp5.types.samples import Samples

import numpy as np
import numpy.typing as npt

def pixel_errors(reconstructions: Samples, originals: Samples, threshold: float = 0.5) -> npt.NDArray[np.int64]:
    """
    Wrong pixels per pattern: each output is binarized with `threshold` and compared with the
    original binary pattern.

    Returns:
        Vector (samples,) with the count of pixels that differ.
    """
    binarized = reconstructions >= threshold
    return np.sum(binarized != (originals >= threshold), axis=1)
