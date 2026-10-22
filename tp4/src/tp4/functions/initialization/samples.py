from tp4.types.float import Float
from tp4.types.samples import Samples

import numpy.typing as npt
from numpy.random import RandomState

# Weights start as random training samples: every neuron starts close to the
# data, which avoids dead units that never win
def samples_initialization(random: RandomState, samples: Samples, neurons: int) -> npt.NDArray[Float]:
    """neurons random rows of samples, without repetition while there are enough samples."""
    rows = random.choice(len(samples), size=neurons, replace=neurons > len(samples))
    return samples[rows].copy()
