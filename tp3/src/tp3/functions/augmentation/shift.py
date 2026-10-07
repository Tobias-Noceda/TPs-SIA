from tp3.types.float import Float

import numpy as np
import numpy.typing as npt
from numpy.random import RandomState

def random_shift(images: npt.NDArray[Float], random: RandomState, max_shift: int = 1, side: int = 28) -> npt.NDArray[Float]:
    """
    Moves every flattened `side × side` image by its own random (dx, dy) ∈ [−max_shift, max_shift]²,
    filling the uncovered border with 0 (background).
    """
    if max_shift == 0:
        return images

    count = len(images)
    padded = np.pad(images.reshape(count, side, side), ((0, 0), (max_shift, max_shift), (max_shift, max_shift)))
    dx, dy = random.randint(-max_shift, max_shift + 1, size=(2, count))

    shifted = np.empty((count, side, side), dtype=images.dtype)
    for x in range(-max_shift, max_shift + 1):
        for y in range(-max_shift, max_shift + 1):
            selected = (dx == x) & (dy == y)
            shifted[selected] = padded[selected, max_shift - y:max_shift - y + side, max_shift - x:max_shift - x + side]

    return shifted.reshape(count, -1)
