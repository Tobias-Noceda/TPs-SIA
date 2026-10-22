from tp4.types.float import Float
from tp4.types.grid import GridDistances

import numpy as np
import numpy.typing as npt

_TOLERANCE = 1e-9

# Topographic error: how well the map preserves the topology of the input space. A sample whose two
# closest neurons aren't adjacent on the map means the map folds there.
def topographic_error(scores: npt.NDArray[Float], grid_distances: GridDistances) -> Float:
    """
    Share of samples whose best and second best neurons are farther than 1 on the map.

    scores: dissimilarity (P, neurons) from a SimilarityFunction. Adjacent = map distance ≤ 1
    (4 neighbors on a rectangular grid, 6 on a hexagonal one). Requires at least 2 neurons.
    """
    best = np.argsort(scores, axis=1)[:, :2]
    return np.float64((grid_distances[best[:, 0], best[:, 1]] > 1 + _TOLERANCE).mean())
