from tp5.types.samples import Samples

import numpy as np
from numpy.random import RandomState

# @link https://en.wikipedia.org/wiki/Gaussian_noise
def gaussian_noise(random: RandomState, samples: Samples, level: float) -> Samples:
    """Adds N(0, level²) to every pixel and clips to [0, 1]. Works for grayscale and RGB."""
    return np.clip(samples + random.normal(0.0, level, size=samples.shape), 0.0, 1.0)
