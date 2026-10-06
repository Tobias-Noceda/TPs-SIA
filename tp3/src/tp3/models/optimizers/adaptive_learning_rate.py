from dataclasses import dataclass, field
from typing import override

from tp3.interfaces.optimizer import Optimizer
from tp3.types.float import Float

# Gradient descent with an adaptive learning rate (theory, eq. 8.13), adjusted once per epoch from the
# change of the training loss ΔE:
#   Δη = +a   if ΔE < 0 for k consecutive epochs (and every following decrease while it holds)
#   Δη = -bη  if ΔE > 0 (the loss started growing)
#   Δη = 0    otherwise
# @link https://en.wikipedia.org/wiki/Learning_rate#Adaptive_learning_rate
@dataclass
class AdaptiveLearningRate(Optimizer):
    # Additive increase of η
    a: float
    # Fraction of η removed when the loss grows
    b: float = 0.5
    # Consecutive decreasing epochs required before increasing η
    k: int = 3

    # Current η, starts at the configured learning rate on the first step
    learning_rate: Float | None = field(default=None, init=False)
    previous_loss: Float | None = field(default=None, init=False, repr=False)
    decreasing: int = field(default=0, init=False, repr=False)

    @override
    def step(self, learning_rate, params, gradients) -> None:
        if self.learning_rate is None:
            self.learning_rate = learning_rate

        for param, gradient in zip(params, gradients):
            param -= self.learning_rate * gradient

    @override
    def end_epoch(self, loss) -> None:
        if self.previous_loss is not None and self.learning_rate is not None:
            if loss < self.previous_loss:
                self.decreasing += 1
                if self.decreasing >= self.k:
                    self.learning_rate += self.a
            elif loss > self.previous_loss:
                self.decreasing = 0
                self.learning_rate -= self.b * self.learning_rate
        self.previous_loss = loss
