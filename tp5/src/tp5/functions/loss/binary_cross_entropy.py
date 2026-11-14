from tp5.types.float import Float

import numpy as np
import numpy.typing as npt

# Keeps log() and the division of the derivative finite when an output saturates at 0 or 1
_CLIP = 1e-7

# @link https://en.wikipedia.org/wiki/Cross-entropy#Cross-entropy_loss_function_and_logistic_regression
def binary_cross_entropy(arr1: npt.NDArray[Float], arr2: npt.NDArray[Float]):
    """
    Mean binary cross entropy between the outputs (arr1, in (0, 1)) and the expected values (arr2, in {0, 1}):
    -mean[ζ log O + (1 - ζ) log(1 - O)].
    """
    o = np.clip(arr1, _CLIP, 1 - _CLIP)
    return np.float64(-np.mean(arr2 * np.log(o) + (1 - arr2) * np.log(1 - o)))

def binary_cross_entropy_derivative(arr1: npt.NDArray[Float], arr2: npt.NDArray[Float]):
    # Per-sample derivative ∂E/∂O = (O - ζ) / (O (1 - O)), divided by the outputs count like MSE.
    # Times the logistic derivative 2β O (1 - O) it reduces to 2β (O - ζ): no vanishing gradient
    # when a logistic output saturates on the wrong side
    o = np.clip(arr1, _CLIP, 1 - _CLIP)
    return (o - arr2) / (o * (1 - o)) / np.shape(arr1)[-1]
