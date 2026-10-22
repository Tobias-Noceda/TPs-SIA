from typing import Callable

from tp4.types.float import Float
from tp4.types.samples import Samples

import numpy.typing as npt
from numpy.random import RandomState

# Draws the initial weights of every neuron from the training samples: (random, samples (P, n), neurons) → (neurons, n)
type InitializationFunction = Callable[[RandomState, Samples, int], npt.NDArray[Float]]
