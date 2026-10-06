from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from tp3.interfaces.optimizer import Optimizer
from tp3.models.neuron import Neuron
from tp3.models.optimizers.gradient_descent import GradientDescent
from tp3.functions.initialization.uniform import uniform_initialization
from tp3.types.activation_function import ActivationFunction, DerivativeActivationFunction
from tp3.types.float import Float
from tp3.types.weight_initialization import WeightInitialization

import numpy.typing as npt

import numpy as np

@dataclass
class Layer:
    input_size: int
    neurons: Tuple[Neuron, ...]
    optimizer: Optimizer = field(default_factory=GradientDescent)
    weight_initialization: WeightInitialization = uniform_initialization

    bias: npt.NDArray[Float] = field(init=False, repr=False, hash=False)
    weights: npt.NDArray[Float] = field(init=False, repr=False, hash=False)

    # Neurons sharing the same activation, with their column indices, so each function is applied
    # once per group over the whole batch instead of once per neuron and sample
    groups: List[Tuple[Neuron, npt.NDArray[np.intp]]] = field(init=False, repr=False, hash=False)

    def __post_init__(self) -> None:
        self.weights = self.weight_initialization(self.input_size, len(self.neurons))
        self.bias = np.random.uniform(-0.5, 0.5, size=len(self.neurons))

        columns: Dict[Tuple[ActivationFunction, DerivativeActivationFunction], List[int]] = {}
        representatives: Dict[Tuple[ActivationFunction, DerivativeActivationFunction], Neuron] = {}
        for i, neuron in enumerate(self.neurons):
            key = (neuron.activation_function, neuron.derivative_activation_function)
            columns.setdefault(key, []).append(i)
            representatives.setdefault(key, neuron)

        self.groups = [(representatives[key], np.array(cols)) for key, cols in columns.items()]

    def fowards(self, data: npt.NDArray[Float]) -> Tuple[npt.NDArray[Float], npt.NDArray[Float]]:
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
        learning_rate: Float,
        data: npt.NDArray[Float],
        delta: npt.NDArray[Float],
    ):
        # data.T @ delta = Σ_μ outer(V^μ, δ^μ). Averaged over the batch so the effective
        # learning rate does not depend on the batch size (batch of 1 = online update)
        batch_size = data.shape[0]
        gradients = ((data.T @ delta) / batch_size, delta.mean(axis=0))
        self.optimizer.step(learning_rate, (self.weights, self.bias), gradients)
