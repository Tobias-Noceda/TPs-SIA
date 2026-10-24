from typing import Callable

from tp4.types.float import Float

import numpy.typing as npt

# Picks the sign of each eigenvector (columns of a (p, p) matrix). An eigenvector v and −v are both
# valid, so the sign returned by the solver is arbitrary.
type OrientationFunction = Callable[[npt.NDArray[Float]], npt.NDArray[Float]]
