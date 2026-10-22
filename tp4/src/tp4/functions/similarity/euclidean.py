from tp4.types.float import Float
from tp4.types.samples import Samples

import numpy as np
import numpy.typing as npt

# Euclidean distance ‖Xp - Wj‖
# @link https://en.wikipedia.org/wiki/Euclidean_distance
def euclidean_similarity(weights: Samples, samples: Samples) -> npt.NDArray[Float]:
    """Distance between every sample (P, n) and every neuron's weights (neurons, n): (P, neurons)."""
    return np.linalg.norm(samples[:, None, :] - weights[None, :, :], axis=2)
