from tp5.types.float import Float
from tp5.types.learning_rate_function import LearningRateFunction

import numpy as np
import numpy.typing as npt

# @link https://arxiv.org/abs/1412.6980
def adam_learning_rate(learning_rate: float, beta1: float, beta2: float, epsilon: float) -> LearningRateFunction:
    """
    Adam: moving averages of the gradient (m) and of its element-wise square (v), bias corrected
    by 1 - βᵗ because both start at 0. t counts calls (one per batch), it is never reset per epoch.

        θ_t = θ_{t-1} - α m̂_t / (√v̂_t + ε)

    Args:
        learning_rate (float): The step size α (0.001 in the paper).
        beta1 (float): Decay rate of the first moment (0.9 in the paper).
        beta2 (float): Decay rate of the second moment (0.999 in the paper).
        epsilon (float): Avoids dividing by zero (1e-8 in the paper).

    Returns:
        LearningRateFunction: (gradient, error) -> α m̂ / (√v̂ + ε).
    """
    m: npt.NDArray[Float] | None = None
    v: npt.NDArray[Float] | None = None
    # Running β₁ᵗ and β₂ᵗ instead of a counter t, to avoid recomputing the powers every step
    beta1_t = 1.0
    beta2_t = 1.0

    def step(gradient: npt.NDArray[Float], error: Float) -> npt.NDArray[Float]:
        nonlocal m, v, beta1_t, beta2_t
        if m is None or v is None:
            m = np.zeros_like(gradient, dtype=np.float64)
            v = np.zeros_like(gradient, dtype=np.float64)

        beta1_t *= beta1
        beta2_t *= beta2

        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient ** 2

        m_hat = m / (1 - beta1_t)
        v_hat = v / (1 - beta2_t)

        return learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    return step
