from typing import Tuple

from tp4.functions.noise.nearest import nearest_entries
from tp4.types.float import Float
from tp4.types.pattern import Pattern

import numpy as np
from numpy.random import RandomState

# Ink or coffee drop, marker blot: a dark disc over the print
def stain_noise(random: RandomState, pattern: Pattern, shape: Tuple[int, int], level: Float) -> Pattern:
    """
    Copy of the pattern where the round(level · N) pixels closest (Euclidean, a disc clipped by the
    borders) to a random center are black (+1): (N,). `level` is the fraction of the image stained;
    pixels already black don't change.
    """
    noisy = pattern.copy()
    rows, columns = np.indices(shape)
    center = (random.randint(shape[0]), random.randint(shape[1]))
    distance = ((rows - center[0]) ** 2 + (columns - center[1]) ** 2).ravel()
    noisy[nearest_entries(random, distance, int(round(level * len(pattern))))] = 1
    return noisy
