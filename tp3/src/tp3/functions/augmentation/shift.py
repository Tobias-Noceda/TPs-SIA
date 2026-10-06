from tp3.types.float import Float

import numpy as np
import numpy.typing as npt

# Data augmentation by random translations (the label of a digit does not change when it moves a
# couple of pixels). Used e.g. by Simard et al., "Best practices for CNNs applied to visual document analysis"
# @link https://doi.org/10.1109/ICDAR.2003.1227801
def random_shift(images: npt.NDArray[Float], max_shift: int, side: int = 28) -> npt.NDArray[Float]:
    """
    Moves every flattened `side × side` image by its own random (dx, dy) ∈ [−max_shift, max_shift]²,
    filling the uncovered border with 0 (background).
    """
    if max_shift == 0:
        return images
    count = len(images)
    padded = np.pad(images.reshape(count, side, side), ((0, 0), (max_shift, max_shift), (max_shift, max_shift)))
    dx, dy = np.random.randint(-max_shift, max_shift + 1, size=(2, count))
    shifted = np.empty((count, side, side), dtype=images.dtype)
    # One slice per distinct shift (at most (2k + 1)² of them) instead of one per image
    for x in range(-max_shift, max_shift + 1):
        for y in range(-max_shift, max_shift + 1):
            selected = (dx == x) & (dy == y)
            shifted[selected] = padded[selected, max_shift - y:max_shift - y + side, max_shift - x:max_shift - x + side]
    return shifted.reshape(count, side * side)
