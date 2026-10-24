from tp4.types.float import Float
from tp4.types.pattern import Pattern

from numpy.random import RandomState

# Noise of a binary pattern: a fraction of its entries changes sign
def flip_noise(random: RandomState, pattern: Pattern, level: Float) -> Pattern:
    """Copy of the pattern with exactly round(level · N) distinct random entries flipped: (N,)."""
    noisy = pattern.copy()
    flipped = random.choice(len(pattern), size=int(round(level * len(pattern))), replace=False)
    noisy[flipped] *= -1
    return noisy
