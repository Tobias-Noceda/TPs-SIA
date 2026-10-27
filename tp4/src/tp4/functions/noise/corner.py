from typing import Tuple

from tp4.functions.noise.nearest import nearest_entries
from tp4.types.float import Float
from tp4.types.pattern import Pattern

import numpy as np
from numpy.random import RandomState

# Torn, broken or folded-away corner of the paper: that area reads as white
def corner_noise(random: RandomState, pattern: Pattern, shape: Tuple[int, int], level: Float) -> Pattern:
    """
    Copy of the pattern where the round(level · N) pixels closest (L1, a triangle) to a random corner
    are white (−1): (N,). `level` is the fraction of the image torn off; pixels already white don't change.
    """
    noisy = pattern.copy()
    rows, columns = np.indices(shape)
    corner = (random.randint(2) * (shape[0] - 1), random.randint(2) * (shape[1] - 1))
    distance = (np.abs(rows - corner[0]) + np.abs(columns - corner[1])).ravel()
    noisy[nearest_entries(random, distance, int(round(level * len(pattern))))] = -1
    return noisy
