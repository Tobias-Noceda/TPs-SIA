from dataclasses import dataclass
from typing import Tuple, override

from tp3.interfaces.perceptron import Perceptron

from tp3.types.activation_function import ActivationFunction
from tp3.types.float import Float
from tp3.types.perceptron_input import PerceptronInput

import numpy.typing as npt

@dataclass(frozen=True)
class SimplePerceptron(Perceptron):
   weights: npt.NDArray[Float]
   activation_function: ActivationFunction

   @override
   def train(self, inputs: Tuple[PerceptronInput, ...], expected: Tuple[Float, ...], epochs: int):
      # Implement the training logic here
      pass

   @override
   def test(self, inputs: Tuple[PerceptronInput, ...], expected: Tuple[Float, ...]):
      # Implement the testing logic here
      pass

   @override
   def predict(self, inputs: Tuple[PerceptronInput, ...]):
      # Implement the prediction logic here
      pass
