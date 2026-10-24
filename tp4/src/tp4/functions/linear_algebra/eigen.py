from typing import Tuple

from tp4.types.float import Float

import numpy as np
import numpy.typing as npt

def sorted_eigen(matrix: npt.NDArray[Float]) -> Tuple[npt.NDArray[Float], npt.NDArray[Float]]:
    """
    Eigenvalues and eigenvectors of a symmetric matrix, sorted from the largest eigenvalue to the smallest.

    Returns (λ (p,), V (p, p)) with V[:, i] the unit eigenvector of λ[i]. Requires a symmetric matrix
    (e.g. a covariance or correlation matrix): its eigenvalues are real and its eigenvectors orthonormal.
    """
    # eigh is the solver for symmetric matrices and returns the eigenvalues in ascending order
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]
