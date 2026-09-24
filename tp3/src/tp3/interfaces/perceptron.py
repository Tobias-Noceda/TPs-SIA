from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy.typing as npt

from tp3.types.epoch_result import EpochResult
from tp3.types.float import Float
from tp3.types.loss_function import LossFunction
from tp3.types.perceptron_input import PerceptronInput

@dataclass(frozen=True)
class Perceptron(ABC):
    learning_rate: Float
    loss_function: LossFunction
    weights: npt.NDArray[Float]

    @abstractmethod
    def train(self, inputs: Tuple[PerceptronInput, ...], expected: Tuple[Float, ...], epochs: int) -> Tuple[EpochResult, ...]:
        pass

    @abstractmethod
    def test(self, inputs: Tuple[PerceptronInput, ...], expected: Tuple[Float, ...]) -> Float:
        pass

    @abstractmethod
    def predict(self, inputs: Tuple[PerceptronInput, ...]) -> Float:
        pass
