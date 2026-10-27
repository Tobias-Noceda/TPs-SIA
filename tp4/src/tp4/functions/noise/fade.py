from typing import Tuple

from tp4.types.float import Float
from tp4.types.pattern import Pattern

import numpy as np
from numpy.random import RandomState

# Low toner / faded or sun-bleached ink: black gets grayer until the scanner reads it as white
def fade_noise(random: RandomState, pattern: Pattern, shape: Tuple[int, int], level: Float) -> Pattern:
    """
    Copy of the pattern with exactly round(level · #black) distinct random black entries (+1) turned
    white (−1): (N,). `level` is the fraction of the black pixels that fade; white never changes.
    """
    noisy = pattern.copy()
    black = np.flatnonzero(pattern > 0)
    faded = random.choice(black, size=int(round(level * len(black))), replace=False)
    noisy[faded] = -1
    return noisy
