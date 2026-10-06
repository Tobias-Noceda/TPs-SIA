from typing import Callable

from tp3.types.float import Float

import numpy.typing as npt
from numpy.random import RandomState

type InitializationFunction = Callable[[RandomState, int, int], npt.NDArray[Float]]
