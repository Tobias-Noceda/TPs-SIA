from tp3.types.float import Float

import numpy as np
import numpy.typing as npt
from numpy.random import RandomState

# Glorot/Xavier uniform: keeps the variance of h about the same across layers, so sigmoid-like
# activations start in their linear zone instead of saturated
# @link https://proceedings.mlr.press/v9/glorot10a.html
def xavier_initialization(random: RandomState, inputs: int, neurons: int) -> npt.NDArray[Float]:
    """U(−r, r) with r = √(6 / (inputs + neurons))."""
    limit = np.sqrt(6 / (inputs + neurons))
    return random.uniform(-limit, limit, size=(inputs, neurons))
