from tp4.types.float import Float
from tp4.types.samples import Samples

import numpy.typing as npt

# Sample correlation = covariance of the standardized variables
# @link https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
def correlation_matrix(standardized: Samples) -> npt.NDArray[Float]:
    """
    Zᵀ Z / P for a (P, p) matrix Z whose columns have mean 0 and standard deviation 1: (p, p).

    Z must be standardized with the population deviation (as fit_z_score does), so that the diagonal is 1.
    """
    return standardized.T @ standardized / len(standardized)
