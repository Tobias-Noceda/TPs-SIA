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
   next_layer_size: int

   bias: npt.NDArray[Float] = field(init=False, repr=False, hash=False)
   weights: npt.NDArray[Float] = field(init=False, repr=False, hash=False)

   def __post_init__(self) -> None:
      self.weights = np.random.uniform(-0.5, 0.5, size=(len(self.perceptrons), self.next_layer_size))
      self.bias = np.random.uniform(-0.5, 0.5, size=self.next_layer_size)

   def fowards(self, data: npt.NDArray[Float]):
      h_us = (self.weights @ data) + self.bias

      activations: List[Float] = []
      derivatives: List[Float] = []
      for i, perceptron in enumerate(self.perceptrons):
         activations.append(perceptron.activation_function(h_us[i]))
         # NOTE: The derivatives are always calculated, even outside of training
         derivatives.append(perceptron.derivative_activation_function(h_us[i]))

      return np.array(activations), np.array(derivatives)

   def backwards(
      self,
      data: npt.NDArray[Float],
      learning_rate: Float,
      derivatives: npt.NDArray[Float],
      next_w: npt.NDArray[Float],
      next_delta: npt.NDArray[Float]
   ):
      old_weights = np.copy(self.weights)
      delta = (next_w.T @ next_delta) * derivatives

      self.weights -= learning_rate * (delta @ data.T)
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
         Layer(perceptrons, 3 if next is None else len(next)) # TODO: hardcoded output size
         for perceptrons, next in zip_longest(self.perceptrons, self.perceptrons[1::])
      ]

   @override
   def train(self, inputs, expected, epochs):
      if len(inputs) != len(expected):
         raise ValueError("The number of inputs and expected outputs must be the same.")

      self.__validate_parameters(inputs)

      results: List[EpochResult] = []
      for epoch in trange(epochs):
         for u, data in enumerate(inputs):
            expect = expected[u]

            outputs = [data]
            activation_derivatives: List[npt.NDArray[Float]] = []

            for layer in self.layers:
               activations, derivative = layer.fowards(outputs[-1])

               outputs.append(activations)
               activation_derivatives.append(derivative)

            prediction = outputs[-1]

            loss = self.loss_function(prediction, expect)
            loss_prime = self.derivative_loss_function(prediction, expect)

            next_w = self.layers[-1].weights
            next_delta = loss_prime * activation_derivatives[-1]

            for layer, ingress, derivatives in zip(
               reversed(self.layers),
               reversed(outputs[:-1]),
               reversed(activation_derivatives),
            ):
               next_w, next_delta = layer.backwards(ingress, self.learning_rate, derivatives, next_w, next_delta)

            results.append((epoch, loss))

      return tuple(results)

   @override
   def test(self, inputs, expected):
      return self.loss_function(np.array([self.predict(inputs)]), np.array([expected]))

   @override
   def predict(self, inputs):
      return tuple([
         self.single_predict(data)
         for data in inputs
      ])

   def single_predict(self, data: npt.NDArray[Float]):
      for layer in self.layers:
         data, _ = layer.fowards(data)
      return data

   def __validate_parameters(self, inputs: Tuple[PerceptronInput, ...]):
      input_dim = len(inputs[0])
      for i in range(len(inputs)):
         if len(inputs[i]) != input_dim:
            raise ValueError("The number of input features must match the number of weights.")

      if self.learning_rate <= 0:
         raise ValueError("The learning rate must be a positive value.")
