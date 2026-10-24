from tp4.types.float import Float

import numpy as np
import numpy.typing as npt

# Kaiser criterion: keep the components that explain more variance than a single standardized variable
# (eigenvalue > 1). Only meaningful for a correlation matrix, where every variable has variance 1.
# @link https://en.wikipedia.org/wiki/Factor_analysis#Criteria_for_determining_the_number_of_factors
def kaiser_components(eigenvalues: npt.NDArray[Float]) -> int:
    """Number of eigenvalues > 1 (at least 1)."""
    return max(1, int(np.sum(eigenvalues > 1)))
