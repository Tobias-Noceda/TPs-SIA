from typing import Callable, Tuple

from tp4.types.float import Float
from tp4.types.pattern import Pattern

from numpy.random import RandomState

# Noise of a flat pattern (N,) whose image has shape (rows, cols): spatial noises need the grid
type NoiseFunction = Callable[[RandomState, Pattern, Tuple[int, int], Float], Pattern]
