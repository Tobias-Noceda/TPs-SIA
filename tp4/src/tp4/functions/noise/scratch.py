from typing import Tuple

from tp4.types.float import Float
from tp4.types.pattern import Pattern

import numpy as np
import numpy.typing as npt
from numpy.random import RandomState

def line_entries(random: RandomState, shape: Tuple[int, int]) -> npt.NDArray[np.intp]:
    """Flat indices of a 1-px straight line across the image (random angle and point), in order along it."""
    angle = random.uniform(0, np.pi)
    point = random.uniform(0, shape[0]), random.uniform(0, shape[1])
    # Half-pixel steps over the whole diagonal: no gaps, duplicates removed keeping the order
    reach = np.hypot(*shape)
    t = np.arange(-reach, reach, 0.5)
    rows = np.floor(point[0] + t * np.sin(angle)).astype(int)
    columns = np.floor(point[1] + t * np.cos(angle)).astype(int)
    inside = (rows >= 0) & (rows < shape[0]) & (columns >= 0) & (columns < shape[1])
    entries = rows[inside] * shape[1] + columns[inside]
    _, first = np.unique(entries, return_index=True)
    return entries[np.sort(first)]

# Scratched print: straight 1-px lines scraping the ink off
def scratch_noise(random: RandomState, pattern: Pattern, shape: Tuple[int, int], level: Float) -> Pattern:
    """
    Copy of the pattern with random straight 1-px lines turned white (−1) until exactly
    round(level · N) pixels are covered (the last line is cut along its direction): (N,).
    `level` is the fraction of the image scratched; pixels already white don't change.
    """
    noisy = pattern.copy()
    target = int(round(level * len(pattern)))
    covered = np.zeros(len(pattern), dtype=bool)
    count = 0
    while count < target:
        line = line_entries(random, shape)
        new = line[~covered[line]][:target - count]
        covered[new] = True
        count += len(new)
    noisy[covered] = -1
    return noisy
