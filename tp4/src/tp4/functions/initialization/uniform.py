from tp4.types.float import Float
from tp4.types.samples import Samples

import numpy.typing as npt
from numpy.random import RandomState

# Uniform random weights, bounded to the range of each feature so they live in the
# data's space. Some neurons may still end up far from every sample and never win (dead units)
def uniform_initialization(random: RandomState, samples: Samples, neurons: int) -> npt.NDArray[Float]:
    """U(min, max) of each feature: (neurons, n)."""
    return random.uniform(samples.min(axis=0), samples.max(axis=0), size=(neurons, samples.shape[1]))
