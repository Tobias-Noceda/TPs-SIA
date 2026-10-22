from tp4.types.float import Float

import numpy as np
import numpy.typing as npt

def _unit_length(inputs: npt.NDArray[Float]) -> npt.NDArray[Float]:
    norm = np.linalg.norm(inputs, axis=1, keepdims=True)
    return inputs / np.where(norm == 0, 1.0, norm)

# Unit length scaling: each sample divided by its 2-norm.
# Works per sample, so nothing is fitted and it can't be undone.
# @link https://en.wikipedia.org/wiki/Feature_scaling#Unit_vector_normalization
def unit_length_normalization(inputs: npt.NDArray[Float]):
    """Return (normalize, None): samples are scaled to norm 1, zero vectors are left as is."""
    return _unit_length, None
