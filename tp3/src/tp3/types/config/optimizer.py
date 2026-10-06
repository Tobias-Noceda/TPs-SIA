import enum

from functools import partial
from typing import Callable, NotRequired, TypedDict

from tp3.interfaces.optimizer import Optimizer
from tp3.models.optimizers.adam import Adam
from tp3.models.optimizers.adaptive_learning_rate import AdaptiveLearningRate
from tp3.models.optimizers.gradient_descent import GradientDescent
from tp3.models.optimizers.momentum import Momentum

class OptimizerHyperparameters(TypedDict):
    momentum: NotRequired[float]
    beta1: NotRequired[float]
    beta2: NotRequired[float]
    epsilon: NotRequired[float]
    a: NotRequired[float]
    b: NotRequired[float]
    k: NotRequired[int]

class Optimizers(enum.Enum):
    GRADIENT_DESCENT = "gradient_descent"
    MOMENTUM = "momentum"
    ADAM = "adam"
    ADAPTIVE = "adaptive"

    def factory(self, hyperparameters: OptimizerHyperparameters | None = None) -> Callable[[], Optimizer]:
        """A constructor of fresh optimizers: every layer needs its own state."""
        classes: dict[Optimizers, Callable[..., Optimizer]] = {
            Optimizers.GRADIENT_DESCENT: GradientDescent,
            Optimizers.MOMENTUM: Momentum,
            Optimizers.ADAM: Adam,
            Optimizers.ADAPTIVE: AdaptiveLearningRate,
        }
        return partial(classes[self], **(hyperparameters or {}))
