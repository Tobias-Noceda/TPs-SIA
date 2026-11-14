from tp5.types.float import Float
from tp5.types.learning_rate_function import LearningRateFunction

import numpy.typing as npt

def constant_learning_rate(learning_rate: float) -> LearningRateFunction:
    """
    Plain gradient descent: Δθ = -η ∂E/∂θ.

    Args:
        learning_rate (float): The constant learning rate η.

    Returns:
        LearningRateFunction: (gradient, error) -> η · gradient.
    """
    def step(gradient: npt.NDArray[Float], error: Float) -> npt.NDArray[Float]:
        return learning_rate * gradient

    return step
