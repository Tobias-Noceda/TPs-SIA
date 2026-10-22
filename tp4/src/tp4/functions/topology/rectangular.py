from tp4.types.grid import GridPositions

import numpy as np

# Rectangular grid: R = 1 reaches the 4 neighbors, R = √2 the 8 neighbors
def rectangular_positions(k: int) -> GridPositions:
    """Neuron j at (x, y) = (j % k, j // k)."""
    rows, columns = np.divmod(np.arange(k * k), k)
    return np.column_stack((columns, rows)).astype(np.float64)
