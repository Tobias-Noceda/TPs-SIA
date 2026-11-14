from dataclasses import dataclass, field
from typing import List, Tuple

from tp5.functions.initialization.random import random_initialization
from tp5.models.layer import Layer
from tp5.models.neuron import Neuron
from tp5.types.float import Float
from tp5.types.initialization_function import InitializationFunction
from tp5.types.learning_rate_function import LearningRateFactory
from tp5.types.samples import Samples

import numpy as np
import numpy.typing as npt
from numpy.random import RandomState

# Backpropagation split in forward / backward / step so several networks can be chained
# (encoder → decoder) and trained as one: backward returns ∂E/∂input for the previous network.
# @link https://en.wikipedia.org/wiki/Backpropagation
@dataclass
class MultiLayerPerceptron:
    neurons: List[Tuple[Neuron, ...]]

    input_size: int

    learning_rate: LearningRateFactory

    # Source of the initial weights, shared with the owner so a seed reproduces the whole model
    random: RandomState = field(default_factory=RandomState, repr=False, hash=False)

    initialization: InitializationFunction = random_initialization

    layers: List[Layer] = field(init=False, repr=False, hash=False)

    # Cache of the last forward (inputs of each layer and activation derivatives) and of the last
    # backward (deltas), consumed by backward and step respectively
    _inputs: List[npt.NDArray[Float]] = field(init=False, repr=False, hash=False, default_factory=list)
    _derivatives: List[npt.NDArray[Float]] = field(init=False, repr=False, hash=False, default_factory=list)
    _deltas: List[npt.NDArray[Float]] = field(init=False, repr=False, hash=False, default_factory=list)

    def __post_init__(self) -> None:
        if not self.neurons:
            raise ValueError("A multi layer perceptron needs at least one layer.")

        self.layers = []
        prev_size = self.input_size
        for neurons in self.neurons:
            self.layers.append(Layer(prev_size, neurons, self.learning_rate, self.random, self.initialization))
            prev_size = len(neurons)

    @property
    def output_size(self) -> int:
        return len(self.neurons[-1])

    def forward(self, data: Samples) -> Samples:
        """
        Forward a batch (batch, input_size) keeping what backward needs.

        Returns:
            Outputs (batch, output_size).
        """
        self._inputs = []
        self._derivatives = []
        for layer in self.layers:
            self._inputs.append(data)
            data, derivative = layer.fowards(data)
            self._derivatives.append(derivative)
        return data

    def backward(self, gradient: npt.NDArray[Float]) -> npt.NDArray[Float]:
        """
        Backpropagate ∂E/∂output of the last forward.

        Parameters:
        gradient : ∂E/∂O per sample (batch, output_size).

        Returns:
            ∂E/∂input per sample (batch, input_size), to keep backpropagating through whatever fed this network.
        """
        if not self._inputs:
            raise RuntimeError("backward requires a previous forward.")

        deltas = []
        for layer, derivative in zip(reversed(self.layers), reversed(self._derivatives)):
            delta = gradient * derivative
            deltas.append(delta)
            gradient = layer.delta(delta)
        deltas.reverse()

        self._deltas = deltas
        return gradient

    def step(self, error: Float) -> None:
        """
        Update every layer with the deltas of the last backward.

        Parameters:
        error : Error of the previous epoch (inf on the first one), for the learning rate functions.
        """
        if not self._deltas:
            raise RuntimeError("step requires a previous backward.")

        for layer, data, delta in zip(self.layers, self._inputs, self._deltas):
            layer.update(data, delta, error)
        self._deltas = []

    def predict(self, data: Samples) -> Samples:
        """Forward without touching the training cache."""
        for layer in self.layers:
            data, _ = layer.fowards(data)
        return data

    def parameters(self) -> List[Tuple[npt.NDArray[Float], npt.NDArray[Float]]]:
        """(weights, bias) of each layer, by reference."""
        return [(layer.weights, layer.bias) for layer in self.layers]

    def load_parameters(self, parameters: List[Tuple[npt.NDArray[Float], npt.NDArray[Float]]]) -> None:
        if len(parameters) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} layers of parameters, got {len(parameters)}.")

        for layer, (weights, bias) in zip(self.layers, parameters):
            if weights.shape != layer.weights.shape or bias.shape != layer.bias.shape:
                raise ValueError(
                    f"Parameter shapes {weights.shape}, {bias.shape} do not match the layer "
                    f"{layer.weights.shape}, {layer.bias.shape}."
                )
            layer.weights = np.array(weights, dtype=np.float64)
            layer.bias = np.array(bias, dtype=np.float64)
