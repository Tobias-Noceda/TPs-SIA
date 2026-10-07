from typing import Callable

from tp3.types.float import Float

import numpy.typing as npt
from numpy.random import RandomState

type AugmentationFunction = Callable[[npt.NDArray[Float], RandomState], npt.NDArray[Float]]
