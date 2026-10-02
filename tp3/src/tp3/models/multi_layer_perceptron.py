from dataclasses import dataclass, field
from typing import List, Tuple, override

from tp3.interfaces.perceptron import Perceptron

from tp3.models.simple_perceptron import SimplePerceptron
from tp3.types.epoch_result import EpochResult
from tp3.types.loss_function import LossFunction
from tp3.types.perceptron_input import PerceptronInput
from tp3.types.float import Float

import numpy.typing as npt

import numpy as np
from tqdm import trange

from itertools import zip_longest

@dataclass
class Layer:
   perceptrons: Tuple[SimplePerceptron, ...]
   next_layer_size: int | None

   bias: npt.NDArray[Float] = field(init=False, repr=False, hash=False)
   weights: npt.NDArray[Float] = field(init=False, repr=False, hash=False)

   def __post_init__(self) -> None:
      if self.next_layer_size is not None:
         self.weights = np.random.uniform(-0.5, 0.5, size=(len(self.perceptrons), self.next_layer_size))
         self.bias = np.random.uniform(-0.5, 0.5, size=self.next_layer_size)

   # NOTE: The last layer (output) will throw an error,
   # as it doesn't define weight and bias values
   def fowards(self, data: npt.NDArray[Float]) -> Tuple[npt.NDArray[Float], npt.NDArray[Float]]:
      h_us = (self.weights @ data) + self.bias

      output = []
      derivatives = []
      for i, perceptron in enumerate(self.perceptrons):
         output.append(perceptron.activation_function(h_us[i]))
         derivatives.append(perceptron.derivative_activation_function(h_us[i]))

      return np.array(output), np.array(derivatives)

   def backwards(
      self,
      data: npt.NDArray[Float],
      learning_rate: Float,
      derivatives: npt.NDArray[Float],
      next_w: npt.NDArray[Float],
      next_delta: npt.NDArray[Float]
   ) -> Tuple[npt.NDArray[Float], npt.NDArray[Float]]:
      old_weights = np.array(self.weights)
      delta = (next_w.T @ next_delta) * derivatives

      self.weights -= learning_rate * (delta * data.T)
      self.bias -= learning_rate * delta

      return old_weights, delta

@dataclass
class MultiLayerPerceptron(Perceptron[npt.NDArray[Float]]):
   learning_rate: Float

   loss_function: LossFunction
   derivative_loss_function: LossFunction

   perceptrons: List[Tuple[SimplePerceptron]]
   layers: List[Layer] = field(init=False, repr=False, hash=False)

   def __post_init__(self):
      super().__init__()

      self.layers = [
         Layer(perceptrons, None if next is None else len(next))
         for perceptrons, next in zip_longest(self.perceptrons, self.perceptrons[1::])
      ]

   @override
   def train(self, inputs, expected, epochs):
      if len(inputs) != len(expected):
         raise ValueError("The number of inputs and expected outputs must be the same.")

      self.__validate_parameters(inputs)

      expect = np.array(expected)
      results: List[EpochResult] = []
      for epoch in trange(epochs):
         for u, data in enumerate(inputs):
            predictions = [data]
            activation_derivatives: List[npt.NDArray[Float]] = []

            for layer in self.layers:
               output, derivative = layer.fowards(predictions[-1])

               predictions.append(output)
               activation_derivatives.append(derivative)

            loss = self.loss_function(predictions[-1], expect)
            loss_prime = self.derivative_loss_function(predictions[-1], expect)

            next_d = loss_prime * activation_derivatives[-1]
            next_w = np.array(self.layers[-1].weights)

            for layer, derivatives in zip(self.layers[:-1], activation_derivatives[:-1]):
               next_w, next_d = layer.backwards(data, self.learning_rate, derivatives, next_w, next_d)

            results.append((epoch, loss))

      return tuple(results)

   @override
   def test(self, inputs, expected):
      # Implement the testing logic here
      return np.float64(0)

   @override
   def predict(self, inputs: Tuple[PerceptronInput, ...]):
      # Implement the prediction logic here
      pass

   def __validate_parameters(self, inputs: Tuple[PerceptronInput, ...]):
      input_dim = len(inputs[0])
      for i in range(len(inputs)):
         if len(inputs[i]) != input_dim:
            raise ValueError("The number of input features must match the number of weights.")

      if self.learning_rate <= 0:
         raise ValueError("The learning rate must be a positive value.")
