from dataclasses import dataclass
from typing import Tuple, List, override

from tp3.interfaces.perceptron import Perceptron

from tp3.types.activation_function import ActivationFunction
from tp3.types.derivative_function import DerivativeFunction
from tp3.types.float import Float
from tp3.types.loss_function import LossFunction
from tp3.types.perceptron_input import PerceptronInput

import numpy as np
import numpy.typing as npt
from tqdm import trange

@dataclass(init=True)
class SimplePerceptron(Perceptron[Float]):

   loss_function: LossFunction
   learning_rate: Float

   activation_function: ActivationFunction
   derivative_activation_function: DerivativeFunction

   bias: Float | None = None
   weights: npt.NDArray[Float] | None = None

   epsilon = np.float64(1e-6)

   @override
   def train(self, inputs, expected, epochs) -> Tuple[tuple[int, Float], ...]:      
      if len(inputs) != len(expected):
         raise ValueError("The number of inputs and expected outputs must be the same.")

      if self.weights is None:
         self.weights = np.random.uniform(-0.5, 0.5, size=len(inputs[0]))
      if self.bias is None:
         self.bias = np.random.uniform(-0.5, 0.5, size=1)[0]

      self.__validate_parameters(inputs)

      if self.weights is None or self.bias is None:
         raise ValueError("Weights and bias must be initialized before training.")
   
      results = []
      for epoch in trange(epochs):
         outputs: List[Float] = []
         for u in range(len(inputs)):
            h_u = np.dot(self.weights, inputs[u]) + self.bias
            
            output = self.activation_function(h_u)
            delta = self.learning_rate * (expected[u] - output) * self.derivative_activation_function(h_u)

            outputs.append(output)

            self.bias = self.bias + delta
            self.weights = self.weights + delta * inputs[u]

         error = self.loss_function(np.array(outputs), np.array(expected))
         results.append((epoch, error))

         if error < self.epsilon:
            break

      return tuple(results)
   
   @override
   def test(self, inputs, expected) -> Float:
      if self.weights is None or self.bias is None:
         raise ValueError("Perceptron must be trained before testing. Weights and bias are not initialized.")

      self.__validate_parameters(inputs)
      
      if len(inputs) != len(expected):
         raise ValueError("The number of inputs and expected outputs must be the same.")

      return self.loss_function(np.array(self.predict(inputs)), np.array([expected]))

   @override
   def predict(self, inputs) -> Tuple[Float, ...]:
      if self.weights is None or self.bias is None:
         raise ValueError("Perceptron must be trained before prediction. Weights and bias are not initialized.")

      self.__validate_parameters(inputs)

      results = []
      for u in range(len(inputs)):
         h_u = np.dot(self.weights, inputs[u]) + self.bias
         results.append(self.activation_function(h_u))

      return tuple(results)

   def __validate_parameters(self, inputs: Tuple[PerceptronInput, ...]):
      input_dim = len(inputs[0])
      for i in range(len(inputs)):
         if len(inputs[i]) != input_dim:
            raise ValueError("The number of input features must match the number of weights.")

      if self.weights is not None and input_dim != len(self.weights):
         raise ValueError("Input dimension does not match initialized weights.")

      if self.learning_rate <= 0:
         raise ValueError("The learning rate must be a positive value.")

      if self.activation_function is None or self.derivative_activation_function is None:
         raise ValueError("Activation function and its derivative must be provided.")
