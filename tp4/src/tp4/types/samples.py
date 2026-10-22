from tp4.types.float import Float

import numpy.typing as npt

# A single sample: vector (features,)
type Sample = npt.NDArray[Float]

# A set of samples, one per row: matrix (sample, feature)
type Samples = npt.NDArray[Float]
