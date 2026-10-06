from abc import ABC, abstractmethod
from typing import Tuple

from tp3.types.float import Float

import numpy.typing as npt

# Weight update rule. Each Layer owns its own instance, so stateful optimizers (momentum, Adam)
# keep one state per parameter of that layer.
class Optimizer(ABC):

    @abstractmethod
    def step(self, learning_rate: Float, params: Tuple[npt.NDArray[Float], ...], gradients: Tuple[npt.NDArray[Float], ...]) -> None:
        """
        Update every parameter IN PLACE with its gradient ∂E/∂param (same shape, same order).
        """
        pass
