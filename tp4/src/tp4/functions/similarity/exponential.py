from tp4.types.float import Float
from tp4.types.samples import Samples

import numpy as np
import numpy.typing as npt

# Exponential similarity e^(-‖Xp - Wj‖²). It grows as the neuron gets closer, so
# it's returned as the dissimilarity 1 - e^(-‖Xp - Wj‖²) to keep "lower = closer". Being a monotone
# transform of the Euclidean distance, it picks the same winner; it saturates at 1 for far neurons.
def exponential_similarity(weights: Samples, samples: Samples) -> npt.NDArray[Float]:
    """1 - e^(-‖Xp - Wj‖²) between every sample (P, n) and every neuron's weights (neurons, n): (P, neurons)."""
    squared = np.sum((samples[:, None, :] - weights[None, :, :]) ** 2, axis=2)
    return 1 - np.exp(-squared)
