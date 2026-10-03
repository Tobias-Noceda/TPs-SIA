from typing import Callable

from tp3.types.float import Float
import numpy.typing as npt

# Function that takes obtained results and expected results and returns a loss value
type LossFunction = Callable[[npt.NDArray[Float], npt.NDArray[Float]], Float]

type DerivativeLossFunction = Callable[[npt.NDArray[Float], npt.NDArray[Float]], Float]
