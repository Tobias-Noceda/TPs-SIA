from dataclasses import dataclass
from typing import Tuple, override

from tp3.interfaces.perceptron import Perceptron

from tp3.types.activation_function import ActivationFunction
from tp3.types.float import Float
from tp3.types.loss_function import LossFunction
from tp3.types.perceptron_input import PerceptronInput

import numpy as np
import numpy.typing as npt
from tqdm import trange

@dataclass(init=True)
class SimplePerceptron(Perceptron):

   loss_function: LossFunction
   learning_rate: Float
   activation_function: ActivationFunction
   weights: npt.NDArray[Float]

   epsilon = np.float64(1e-10)

   @override
   def train(self, inputs, expected, epochs):
      if len(inputs) != len(expected):
         raise ValueError("The number of inputs and expected outputs must be the same.")
      for i in range(len(inputs)):
         if len(inputs[i]) != len(self.weights):
            raise ValueError("The number of input features must match the number of weights.")

      results = []
      for epoch in trange(epochs):
         for u in range(len(inputs)):
            output = self.predict((inputs[u], ))[0]

            self.weights = self.weights + self.learning_rate * (expected[u] - output) * inputs[u]

         error = self.test(inputs, expected)
         results.append((epoch, error))

         if error < self.epsilon:
            break

      return tuple(results)
   
   @override
   def test(self, inputs, expected):
      return self.loss_function(np.array([self.predict(inputs)]), np.array([expected]))

   @override
   def predict(self, inputs):
      if len(inputs) == 0:
         raise ValueError("The input array must not be empty.")
      for i in range(len(inputs)):
         if len(inputs[i]) != len(self.weights):
            raise ValueError("The number of input features must match the number of weights.")

      results = []
      for u in range(len(inputs)):
         h_u = np.dot(self.weights, inputs[u])
         results.append(self.activation_function(h_u))

      return tuple(results)
