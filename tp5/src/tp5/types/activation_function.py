from typing import Callable

from tp5.types.float import Float

import numpy.typing as npt

type ActivationFunction = Callable[[npt.NDArray[Float]], npt.NDArray[Float]]

type DerivativeActivationFunction = Callable[[npt.NDArray[Float]], npt.NDArray[Float]]
