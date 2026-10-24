from tp4.types.float import Float

import numpy as np
import numpy.typing as npt

# Tolerance so neurons exactly at distance R (e.g. R = 1 on a hexagonal grid, computed with √3) count
_TOLERANCE = 1e-9

# Kohonen rule neighborhood: every neuron within R of the winner gets the full update.
# Written ‖n - n_k‖ < R in the theory, but R = 1 must reach the 4 neighbors, so the border is included.
def step_neighborhood(distances: npt.NDArray[Float], radius: Float) -> npt.NDArray[Float]:
    """1 for the neurons at map distance ≤ R from the winner, 0 otherwise."""
    return np.where(distances <= radius + _TOLERANCE, 1.0, 0.0)
