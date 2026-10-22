from tp4.types.float import Float
from tp4.types.grid import GridDistances
from tp4.types.samples import Samples

import numpy as np
import numpy.typing as npt

_TOLERANCE = 1e-9

# Unified distance matrix: high values mark borders between groups of neurons
def u_matrix(weights: Samples, grid_distances: GridDistances) -> npt.NDArray[Float]:
    """Mean Euclidean distance between each neuron's weights and its map neighbors' (map distance ≤ 1): (neurons,)."""
    neighbors = (grid_distances > 0) & (grid_distances <= 1 + _TOLERANCE)
    distances = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
    return (distances * neighbors).sum(axis=1) / neighbors.sum(axis=1)
