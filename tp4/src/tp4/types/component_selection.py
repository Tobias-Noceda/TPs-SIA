from typing import Callable

from tp4.types.float import Float

import numpy.typing as npt

# Number of principal components to keep given the eigenvalues sorted from largest to smallest: (p,) → q
type ComponentSelection = Callable[[npt.NDArray[Float]], int]
