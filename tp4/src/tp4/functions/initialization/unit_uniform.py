from tp4.types.float import Float
from tp4.types.samples import Samples

import numpy.typing as npt
from numpy.random import RandomState

# Uniform random weights in [0, 1), regardless of the samples' range
def unit_uniform_initialization(random: RandomState, samples: Samples, neurons: int) -> npt.NDArray[Float]:
    """U(0, 1): (neurons, n)."""
    return random.uniform(0.0, 1.0, size=(neurons, samples.shape[1]))
