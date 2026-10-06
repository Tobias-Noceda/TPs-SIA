from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Tuple

from tp3.types.classification_function import ClassificationFunction
from tp3.types.float import Float
from tp3.types.perceptron_input import PerceptronInputs
from tp3.types.epoch_result import EpochResult
from tp3.types.test_result import TestResult

import numpy.typing as npt

# Generic perceptron interface (outputs type determined by the implementation).
# Inputs are matrices (samples, features); expected values and predictions hold one OT per row.
@dataclass(init=True)
class Perceptron[OT](ABC):

    @abstractmethod
    def train(self, inputs: PerceptronInputs, expected: npt.NDArray[Float], epochs: int, classification_function: ClassificationFunction[OT] | None, test_function: Callable[[], Float] | None) -> Tuple[EpochResult, ...]:
        pass

    @abstractmethod
    def test(self, inputs: PerceptronInputs, expected: npt.NDArray[Float], classification_function: ClassificationFunction[OT]) -> TestResult:
        pass

    @abstractmethod
    def predict(self, inputs: PerceptronInputs) -> npt.NDArray[Float]:
        pass
