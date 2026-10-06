from typing import Callable

from tp3.types.float import Float

import numpy.typing as npt

# Draws the initial weights of a layer given its (inputs, neurons) shape
type WeightInitialization = Callable[[int, int], npt.NDArray[Float]]
