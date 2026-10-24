from tp4.functions.activation.sign import sign

from tp4.types.float import Float
from tp4.types.pattern import Pattern

import numpy.typing as npt
from numpy.random import RandomState

# Every neuron is updated at once from the previous state: S_i(t + 1) = sgn(Σ_j w_ij S_j(t))
def synchronous_update(weights: npt.NDArray[Float], state: Pattern, random: RandomState) -> Pattern:
    """S(t + 1) = sgn(W · S(t)), keeping S_i(t) where the field is 0: (N,)."""
    return sign(weights @ state, state)
