from tp3.types.float import Float
from tp3.types.learning_rate_function import LearningRateFunction

import numpy as np
import numpy.typing as npt

# @link theory/artificializando.pdf, sección 8.3.1, ecuación 8.12
def momentum_learning_rate(learning_rate: float, alpha: float) -> LearningRateFunction:
    """
    Momentum: Δw(t+1) = -η ∂E/∂w + α Δw(t).

    The velocity is kept as the step to subtract (velocity = -Δw), so velocity = α velocity + η g.

    Args:
        learning_rate (float): The learning rate η.
        alpha (float): The momentum factor α (usually 0.8 or 0.9).

    Returns:
        LearningRateFunction: (gradient, error) -> velocity.
    """
    velocity: npt.NDArray[Float] | None = None

    def step(gradient: npt.NDArray[Float], error: Float) -> npt.NDArray[Float]:
        nonlocal velocity
        if velocity is None:
            velocity = np.zeros_like(gradient, dtype=np.float64)

        velocity = alpha * velocity + learning_rate * gradient
        return velocity

    return step
