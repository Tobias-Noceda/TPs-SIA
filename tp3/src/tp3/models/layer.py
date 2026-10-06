from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from tp3.models.neuron import Neuron
from tp3.types.activation_function import ActivationFunction, DerivativeActivationFunction
from tp3.types.float import Float
from tp3.types.learning_rate_function import LearningRateFactory, LearningRateFunction
from tp3.types.Initialization_function import InitializationFunction

import numpy.typing as npt

import numpy as np
from numpy.random import RandomState

@dataclass
class Layer:
    input_size: int
    neurons: Tuple[Neuron, ...]
    learning_rate: LearningRateFactory
    random: RandomState
    initialization: InitializationFunction

    bias: npt.NDArray[Float] = field(init=False, repr=False, hash=False)
    weights: npt.NDArray[Float] = field(init=False, repr=False, hash=False)

    # One stateful learning rate function per parameter array
    weights_step: LearningRateFunction = field(init=False, repr=False, hash=False)
    bias_step: LearningRateFunction = field(init=False, repr=False, hash=False)

    # Neurons sharing the same activation, with their column indices, so each function is applied
    # once per group over the whole batch instead of once per neuron and sample
    groups: List[Tuple[Neuron, npt.NDArray[np.intp]]] = field(init=False, repr=False, hash=False)

    def __post_init__(self) -> None:
        # The bias is drawn as the weight of one more input (x0 = 1), so it follows the same distribution
        parameters = self.initialization(self.random, self.input_size + 1, len(self.neurons))
        self.weights = parameters[:-1].copy()
        self.bias = parameters[-1].copy()
        self.weights_step = self.learning_rate()
        self.bias_step = self.learning_rate()

        columns: Dict[Neuron, List[int]] = {}
        representatives: Dict[Neuron, Neuron] = {}
        for i, neuron in enumerate(self.neurons):
            columns.setdefault(neuron, []).append(i)
            representatives.setdefault(neuron, neuron)

        self.groups = [(representatives[key], np.array(cols)) for key, cols in columns.items()]

    def fowards(self, data: npt.NDArray[Float]):
        """
        Forward a batch through the layer.

        Parameters:
        data : Matrix (batch, input_size).

        Returns:
            Activations and activation derivatives, both (batch, neurons).
        """
        h_us = (data @ self.weights) + self.bias

        # NOTE: The derivatives are always calculated, even outside of training
        if len(self.groups) == 1:
            neuron, _ = self.groups[0]
            return neuron.activate(h_us), neuron.derivate(h_us)

        activations = np.empty_like(h_us)
        derivatives = np.empty_like(h_us)
        for neuron, cols in self.groups:
            activations[:, cols] = neuron.activate(h_us[:, cols])
            derivatives[:, cols] = neuron.derivate(h_us[:, cols])

        return activations, derivatives

    def delta(self, delta: npt.NDArray[Float]):
        return delta @ self.weights.T

    def update(
        self,
        data: npt.NDArray[Float],
        delta: npt.NDArray[Float],
        error: Float,
    ):
        """
        Apply one optimization step with the batch gradients.

        Parameters:
        data : Inputs of the layer (batch, input_size).
        delta : ∂E/∂h of each neuron (batch, neurons).
        error : Error of the previous epoch (inf on the first one), for the learning rate functions.
        """
        # Averaged over the batch so the effective learning rate does not depend on its size
        batch_size = data.shape[0]
        self.weights -= self.weights_step((data.T @ delta) / batch_size, error)
        self.bias -= self.bias_step(delta.mean(axis=0), error)
