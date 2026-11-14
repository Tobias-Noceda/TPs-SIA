from typing import Callable

from tp5.types.float import Float

import numpy.typing as npt
from numpy.random import RandomState

type InitializationFunction = Callable[[RandomState, int, int], npt.NDArray[Float]]
