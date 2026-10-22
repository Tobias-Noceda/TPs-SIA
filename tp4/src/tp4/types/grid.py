from tp4.types.float import Float

import numpy.typing as npt

# Position (x, y) of every neuron of the map, one per row: matrix (neurons, 2).
# Neuron j sits at row j // k, column j % k of the k × k grid.
type GridPositions = npt.NDArray[Float]

# Distance on the map between every pair of neurons: matrix (neurons, neurons)
type GridDistances = npt.NDArray[Float]
