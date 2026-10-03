from dataclasses import dataclass, field
from typing import List, Tuple

from tp3.models.simple_perceptron import SimplePerceptron
from tp3.types.float import Float

import numpy.typing as npt

import numpy as np

@dataclass
class Layer:
    input_size: int
    perceptrons: Tuple[SimplePerceptron, ...]

    bias: npt.NDArray[Float] = field(init=False, repr=False, hash=False)
    weights: npt.NDArray[Float] = field(init=False, repr=False, hash=False)

    def __post_init__(self) -> None:
        self.weights = np.random.uniform(-0.5, 0.5, size=(len(self.perceptrons), self.input_size))
        self.bias = np.random.uniform(-0.5, 0.5, size=len(self.perceptrons))

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
