from typing import Callable

from tp4.types.pattern import Pattern

from numpy.random import RandomState

type NoiseFunction = Callable[[RandomState, Pattern, float], Pattern]
