from tp3.types.float import Float

import numpy.typing as npt

# A single sample: vector (features,)
type PerceptronInput = npt.NDArray[Float]

# A set of samples, one per row: matrix (samples, features)
type PerceptronInputs = npt.NDArray[Float]
