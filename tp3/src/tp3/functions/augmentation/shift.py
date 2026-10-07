from typing import Tuple

from tp3.types.float import Float

import numpy as np
import numpy.typing as npt
from numpy.random import RandomState

def random_shift(data: npt.NDArray[Float], random: RandomState, shape: Tuple[int, ...], max_shift: Tuple[int, ...]):
    """
    Moves every flattened `side × side` image by its own random (dx, dy) ∈ [−max_shift, max_shift]²,
    filling the uncovered border with 0 (background).
    """
    dimensions = len(shape)
    count = len(data)

    reshaped = data.reshape((count, *shape))

    paddings = [(0, 0)] + [(shift, shift) for shift in max_shift]
    padded = np.pad(reshaped, paddings, mode="constant", constant_values=0)

    sampled_shifts = np.column_stack([
        random.randint(-shift, shift + 1, size=count)
        for shift in max_shift
    ])

    sample_indices = np.arange(count).reshape((count,) + (1,) * dimensions)

    spatial_indices = []
    for dimension, (size, max_displacement) in enumerate(zip(shape, max_shift)):
        indices = (
            np.arange(size)[None, :]
            + max_displacement
            - sampled_shifts[:, dimension][:, None]
        )

        spatial_indices.append(
            indices.reshape(
                (count,) + (1,) * dimension + (size,) + (1,) * (dimensions - dimension - 1)
            )
        )

    shifted = padded[(sample_indices, *spatial_indices)]

    return shifted.reshape(count, -1)
