from tp4.types.float import Float
from tp4.types.samples import Samples

import numpy as np
import numpy.typing as npt

# Quantization error: how well each neuron's weights represent the samples it wins
def quantization_error(samples: Samples, weights: Samples, winners: npt.NDArray[np.intp]) -> Float:
    """Mean Euclidean distance between each sample and its winner's weights."""
    return np.float64(np.linalg.norm(samples - weights[winners], axis=1).mean())
