from tp5.types.samples import Samples

import numpy as np
from numpy.random import RandomState

def flip_noise(random: RandomState, samples: Samples, level: float) -> Samples:
    """Each pixel is inverted (x → 1 − x) with probability `level`. For binary images."""
    mask = random.random_sample(samples.shape) < level
    return np.where(mask, 1 - samples, samples)
