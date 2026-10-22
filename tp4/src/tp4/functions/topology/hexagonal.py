from tp4.types.grid import GridPositions

import numpy as np

# Hexagonal grid: odd rows shifted half a cell so each neuron has 6 neighbors at distance 1
# @link https://www.redblobgames.com/grids/hexagons/
def hexagonal_positions(k: int) -> GridPositions:
    """Neuron j at (x, y) = (j % k + (row % 2) / 2, row · √3/2) with row = j // k."""
    rows, columns = np.divmod(np.arange(k * k), k)
    return np.column_stack((columns + (rows % 2) / 2, rows * np.sqrt(3) / 2)).astype(np.float64)
