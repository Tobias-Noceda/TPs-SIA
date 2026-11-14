from tp5.types.float import Float
from tp5.types.learning_rate_function import LearningRateFunction

import numpy as np
import numpy.typing as npt

def adaptative_learning_rate(learning_rate: float, a: float, b: float, k: int) -> LearningRateFunction:
    """
    Adaptive learning rate, adjusted once per epoch from the change of the error ΔE:

        Δη = +a   if ΔE < 0 for k epochs in a row
        Δη = -bη  if ΔE > 0 for k epochs in a row
        Δη = 0    otherwise

    It is called once per batch with the error of the previous epoch, so an epoch boundary is
    detected when that error changes. Two epochs with exactly the same error look like a single
    one, which is harmless: ΔE = 0 gives Δη = 0 anyway.

    Args:
        learning_rate (float): The initial learning rate η.
        a (float): Additive increase.
        b (float): Multiplicative decrease, in (0, 1).
        k (int): Epochs in a row with the same sign of ΔE before adjusting η.

    Returns:
        LearningRateFunction: (gradient, error) -> η · gradient.
    """
    eta = learning_rate
    previous_error: Float = np.float64(np.inf)
    # Signed streak: +n = n epochs in a row with ΔE > 0, -n = n epochs in a row with ΔE < 0
    streak = 0

    def step(gradient: npt.NDArray[Float], error: Float) -> npt.NDArray[Float]:
        nonlocal eta, previous_error, streak

        if error != previous_error:
            # No ΔE until two epoch errors are known (the first call receives inf)
            if np.isfinite(previous_error) and np.isfinite(error):
                if error < previous_error:
                    streak = streak - 1 if streak < 0 else -1
                else:
                    streak = streak + 1 if streak > 0 else 1

                if streak <= -k:
                    eta += a
                elif streak >= k:
                    eta -= b * eta

            previous_error = error

        return eta * gradient

    return step
