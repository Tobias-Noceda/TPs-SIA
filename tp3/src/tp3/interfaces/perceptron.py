from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

from tp3.types.epoch_result import EpochResult
from tp3.types.float import Float
from tp3.types.perceptron_input import PerceptronInput

# Generic perceptron interface (outputs type determined by the implementation)
@dataclass(init=True)
class Perceptron[OT](ABC):

    @abstractmethod
    def train(self, inputs: Tuple[PerceptronInput, ...], expected: Tuple[OT, ...], epochs: int) -> Tuple[EpochResult, ...]:
        pass

    @abstractmethod
    def test(self, inputs: Tuple[PerceptronInput, ...], expected: Tuple[OT, ...]) -> Float:
        pass

    @abstractmethod
    def predict(self, inputs: Tuple[PerceptronInput, ...]) -> Tuple[OT, ...]:
        pass
