from tp4.types.float import Float
from tp4.types.pattern import Pattern

import numpy as np
import numpy.typing as npt

def sign(h: npt.NDArray[Float], previous: Pattern) -> Pattern:
    """
    Sign activation of a Hopfield neuron.

    Parameters:
    h : Local field Σ_j w_ij S_j of each neuron.
    previous : State of each neuron before the update.

    Returns:
        1 if h > 0, -1 if h < 0 and the previous state if h = 0.
    """
    return np.where(h > 0, 1.0, np.where(h < 0, -1.0, previous))
