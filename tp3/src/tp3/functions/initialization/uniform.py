from tp3.types.float import Float

import numpy as np
import numpy.typing as npt

def uniform_initialization(inputs: int, neurons: int) -> npt.NDArray[Float]:
    """U(−0.5, 0.5) regardless of the layer size (the original initialization)."""
    return np.random.uniform(-0.5, 0.5, size=(inputs, neurons))
