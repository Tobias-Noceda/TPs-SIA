from dataclasses import dataclass, field
from typing import override

from tp4.interfaces.projection import Projection

from tp4.functions.linear_algebra.eigen import sorted_eigen
from tp4.functions.linear_algebra.orientation import max_loading_positive
from tp4.functions.normalization.z_score import fit_z_score, z_score
from tp4.functions.statistics.correlation import correlation_matrix

from tp4.types.component_selection import ComponentSelection
from tp4.types.float import Float
from tp4.types.orientation_function import OrientationFunction
from tp4.types.samples import Samples

import numpy as np
import numpy.typing as npt

# Principal component analysis over the correlation matrix (Hotelling, 1933)
# @link https://en.wikipedia.org/wiki/Principal_component_analysis
@dataclass
class PCA(Projection):
    """
    Principal components of the standardized variables.

    The variables are standardized by the model itself, so fit and project take the raw (P, p) matrix.
    Requires at least 2 samples; a constant column gets a zero row/column in the correlation matrix
    (and a zero eigenvalue) instead of failing.
    """
    selection: ComponentSelection
    orientation: OrientationFunction = max_loading_positive

    # Standardization statistics of the fitted samples, reapplied in project
    mean: npt.NDArray[Float] = field(init=False, repr=False)
    std: npt.NDArray[Float] = field(init=False, repr=False)

    # Correlation matrix (p, p)
    correlation: npt.NDArray[Float] = field(init=False, repr=False)

    # λ from largest to smallest (p,): the variance of each principal component
    eigenvalues: npt.NDArray[Float] = field(init=False, repr=False)

    # V (p, p): column i holds the loadings of the variables on component i
    eigenvectors: npt.NDArray[Float] = field(init=False, repr=False)

    # Number of components q kept by project
    components: int = field(init=False)

    @override
    def fit(self, inputs: Samples) -> None:
        # X with the variables in columns
        inputs = self.__validate_inputs(inputs)
        if len(inputs) < 2:
            raise ValueError("PCA requires at least 2 samples.")

        # Standardize X
        self.mean, self.std = fit_z_score(inputs)
        standardized = z_score(inputs, self.mean, self.std)

        # Correlation matrix Sx
        self.correlation = correlation_matrix(standardized)

        # Eigenvalues and eigenvectors, from the largest eigenvalue to the smallest
        self.eigenvalues, eigenvectors = sorted_eigen(self.correlation)
        self.eigenvectors = self.orientation(eigenvectors)

        # V keeps the eigenvectors of the largest eigenvalues
        self.components = self.selection(self.eigenvalues)

    @override
    def project(self, inputs: Samples) -> Samples:
        """New variables Y = Z V as linear combinations of the standardized originals: (P, q)."""
        return self.scores(inputs)[:, :self.components]

    def scores(self, inputs: Samples) -> Samples:
        """Projection onto every principal component, also the discarded ones: (P, p)."""
        if not hasattr(self, "eigenvectors"):
            raise RuntimeError("The PCA must be fitted first.")

        inputs = self.__validate_inputs(inputs)
        if inputs.shape[1] != len(self.mean):
            raise ValueError(f"Expected {len(self.mean)} variables, got {inputs.shape[1]}.")

        return z_score(inputs, self.mean, self.std) @ self.eigenvectors

    @property
    def explained_variance_ratio(self) -> npt.NDArray[Float]:
        """λᵢ / Σλ (p,): share of the total variance explained by each component (Σλ = p for a correlation matrix)."""
        return self.eigenvalues / np.sum(self.eigenvalues)

    def __validate_inputs(self, inputs: Samples) -> Samples:
        inputs = np.asarray(inputs, dtype=np.float64)

        if inputs.ndim != 2:
            raise ValueError("Inputs must be a (samples, variables) matrix.")

        return inputs
