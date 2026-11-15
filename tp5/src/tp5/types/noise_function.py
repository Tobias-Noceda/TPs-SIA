from typing import Callable

from tp5.types.samples import Samples

from numpy.random import RandomState

# Corrupts a batch of samples (values in [0, 1]) with the given level; returns a new matrix of the same
# shape, never modifies its input.
type NoiseFunction = Callable[[RandomState, Samples, float], Samples]
