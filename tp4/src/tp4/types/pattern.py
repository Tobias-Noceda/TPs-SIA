from tp4.types.float import Float

import numpy.typing as npt

# A binary pattern or state of the network: vector (N,) of ±1
type Pattern = npt.NDArray[Float]

# A set of patterns, one per row: matrix (p, N) of ±1
type Patterns = npt.NDArray[Float]
