from tp5.types.float import Float

import numpy.typing as npt
from numpy.random import RandomState

# Small random values, independent of the layer size (avoids the symmetry
# problem of starting at zero)
def random_initialization(random: RandomState, inputs: int, neurons: int) -> npt.NDArray[Float]:
    """U(-0.5, 0.5)."""
    return random.uniform(-0.5, 0.5, size=(inputs, neurons))