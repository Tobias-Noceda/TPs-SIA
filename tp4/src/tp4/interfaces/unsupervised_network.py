from abc import ABC, abstractmethod
from typing import Tuple

from tp4.types.samples import Samples

import numpy.typing as npt

# Generic unsupervised network: learns from the samples alone (no expected values).
# R is what the implementation reports at the end of each training epoch.
class UnsupervisedNetwork[R](ABC):

    @abstractmethod
    def train(self, inputs: Samples, epochs: int) -> Tuple[R, ...]:
        pass

    @abstractmethod
    def predict(self, inputs: Samples) -> npt.NDArray:
        pass
