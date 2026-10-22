from tp4.types.float import Float

import numpy as np
import numpy.typing as npt

# Gaussian neighborhood: the update fades with the map distance to the winner instead of cutting at R
# @link https://en.wikipedia.org/wiki/Self-organizing_map#Learning_algorithm
def gaussian_neighborhood(distances: npt.NDArray[Float], radius: Float) -> npt.NDArray[Float]:
    """e^(-d² / 2R²). Requires R > 0."""
    return np.exp(-(distances ** 2) / (2 * radius ** 2))
