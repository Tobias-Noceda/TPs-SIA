from dataclasses import dataclass
from typing import override

from tp3.interfaces.optimizer import Optimizer

# Plain gradient descent: Δw = -η ∂E/∂w
# @link https://en.wikipedia.org/wiki/Gradient_descent
@dataclass
class GradientDescent(Optimizer):

    @override
    def step(self, learning_rate, params, gradients) -> None:
        for param, gradient in zip(params, gradients):
            param -= learning_rate * gradient
