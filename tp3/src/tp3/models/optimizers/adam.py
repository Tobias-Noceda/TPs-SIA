from dataclasses import dataclass, field
from typing import List, override

from tp3.interfaces.optimizer import Optimizer
from tp3.types.float import Float

import numpy as np
import numpy.typing as npt

# Adam: per-parameter step from bias-corrected moving averages of the gradient and its square
# @link https://arxiv.org/abs/1412.6980
@dataclass
class Adam(Optimizer):
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    t: int = field(default=0, init=False, repr=False)
    m: List[npt.NDArray[Float]] = field(default_factory=list, init=False, repr=False)
    v: List[npt.NDArray[Float]] = field(default_factory=list, init=False, repr=False)

    @override
    def step(self, learning_rate, params, gradients) -> None:
        if not self.m:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1
        for param, gradient, m, v in zip(params, gradients, self.m, self.v):
            m *= self.beta1
            m += (1 - self.beta1) * gradient
            v *= self.beta2
            v += (1 - self.beta2) * gradient ** 2

            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            param -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
