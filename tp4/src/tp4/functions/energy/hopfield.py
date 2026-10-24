from tp4.types.float import Float
from tp4.types.pattern import Pattern

import numpy as np
import numpy.typing as npt

# Energy of a state: its local minima are the attractors of the network (stored or spurious patterns)
# @link https://en.wikipedia.org/wiki/Hopfield_network#Energy
def hopfield_energy(weights: npt.NDArray[Float], state: Pattern) -> Float:
    """H = −½ Σ_ij w_ij S_i S_j."""
    return np.float64(-0.5 * state @ weights @ state)
