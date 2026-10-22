from tp4.types.float import Float

import numpy.typing as npt

def identity_normalization(inputs: npt.NDArray[Float]):
    """Leave the samples unchanged: return (identity, identity)."""
    return (lambda x: x), (lambda x: x)
