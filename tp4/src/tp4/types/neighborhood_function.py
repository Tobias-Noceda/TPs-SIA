from typing import Callable

from tp4.types.float import Float

import numpy.typing as npt

# Update strength of every neuron given its map distance to the winner and the radius R:
# (distances (neurons,), R) → (neurons,) in [0, 1]. The winner (distance 0) always gets 1.
type NeighborhoodFunction = Callable[[npt.NDArray[Float], Float], npt.NDArray[Float]]
