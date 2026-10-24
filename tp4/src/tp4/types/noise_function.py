from typing import Callable

from tp4.types.float import Float
from tp4.types.pattern import Pattern

from numpy.random import RandomState

type NoiseFunction = Callable[[RandomState, Pattern, Float], Pattern]
