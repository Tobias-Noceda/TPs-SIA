from tp4.types.float import Float

import numpy as np
import numpy.typing as npt
from numpy.random import RandomState

def nearest_entries(random: RandomState, distance: npt.NDArray[Float], k: int) -> npt.NDArray[np.intp]:
    """Indices of the k entries with the smallest distance, ties broken at random (exact count, compact shape)."""
    return np.lexsort((random.random_sample(len(distance)), distance))[:k]
