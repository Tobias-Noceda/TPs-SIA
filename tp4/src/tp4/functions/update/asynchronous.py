from tp4.functions.activation.sign import sign

from tp4.types.float import Float
from tp4.types.pattern import Pattern

import numpy.typing as npt
from numpy.random import RandomState

# One neuron at a time, in random order, each one seeing the latest state: the energy never increases
# @link https://en.wikipedia.org/wiki/Hopfield_network#Updating
def asynchronous_update(weights: npt.NDArray[Float], state: Pattern, random: RandomState) -> Pattern:
    """One sweep over the N neurons in random order: (N,)."""
    state = state.copy()
    for i in random.permutation(len(state)):
        state[i] = sign(weights[i] @ state, state[i])
    return state
