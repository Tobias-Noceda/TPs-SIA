from dataclasses import dataclass, field
from typing import List, Tuple

from tp3.models.neuron import Neuron
from tp3.types.float import Float

import numpy.typing as npt

import numpy as np

@dataclass
class Layer:
    input_size: int
    neurons: Tuple[Neuron, ...]

    bias: npt.NDArray[Float] = field(init=False, repr=False, hash=False)
    weights: npt.NDArray[Float] = field(init=False, repr=False, hash=False)

    def __post_init__(self) -> None:
        self.weights = np.random.uniform(-0.5, 0.5, size=(self.input_size, len(self.neurons)))
        self.bias = np.random.uniform(-0.5, 0.5, size=len(self.neurons))

    def fowards(self, data: npt.NDArray[Float]):
        h_us = (data @ self.weights) + self.bias

        activations: List[Float] = []
        derivatives: List[Float] = []
        for i, neuron in enumerate(self.neurons):
            activations.append(neuron.activate(h_us[i]))
            # NOTE: The derivatives are always calculated, even outside of training
            derivatives.append(neuron.derivate(h_us[i]))

        return np.array(activations), np.array(derivatives)

    def delta(self, delta: npt.NDArray[Float]):
        return delta @ self.weights.T

    def update(
        self,
        learning_rate: Float,
        data: npt.NDArray[Float],
        delta: npt.NDArray[Float],
    ):
        self.weights -= learning_rate * np.outer(data.T, delta)
        self.bias -= learning_rate * delta
