from typing import Callable

from tp4.types.float import Float
from tp4.types.pattern import Pattern

import numpy.typing as npt
from numpy.random import RandomState

# One iteration of the Hopfield dynamics: (weights (N, N), state S(t), random) → S(t + 1)
type UpdateFunction = Callable[[npt.NDArray[Float], Pattern, RandomState], Pattern]
