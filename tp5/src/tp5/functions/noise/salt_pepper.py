from tp5.types.samples import Samples

import numpy as np
from numpy.random import RandomState

# @link https://en.wikipedia.org/wiki/Salt-and-pepper_noise
def salt_pepper_noise(random: RandomState, samples: Samples, level: float) -> Samples:
    """Each pixel, with probability `level`, is replaced by 0 or 1 (equally likely)."""
    mask = random.random_sample(samples.shape) < level
    values = (random.random_sample(samples.shape) < 0.5).astype(np.float64)
    return np.where(mask, values, samples)
