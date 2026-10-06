from dataclasses import dataclass, field
from typing import List, override

from tp3.interfaces.optimizer import Optimizer
from tp3.types.float import Float

import numpy as np
import numpy.typing as npt

# Gradient descent with momentum: Δw(t) = -η ∂E/∂w + α Δw(t-1)
# @link https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum
@dataclass
class Momentum(Optimizer):
    # α: fraction of the previous update carried over
    momentum: float = 0.9

    velocities: List[npt.NDArray[Float]] = field(default_factory=list, init=False, repr=False)

    @override
    def step(self, learning_rate, params, gradients) -> None:
        if not self.velocities:
            self.velocities = [np.zeros_like(p) for p in params]

        for param, gradient, velocity in zip(params, gradients, self.velocities):
            velocity *= self.momentum
            velocity -= learning_rate * gradient
            param += velocity
