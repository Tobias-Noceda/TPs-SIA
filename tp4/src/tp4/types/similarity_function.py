from typing import Callable

from tp4.types.float import Float
from tp4.types.samples import Samples

import numpy.typing as npt

# Dissimilarity between every sample and every neuron's weights: (weights (neurons, n), samples (P, n)) → (P, neurons).
# Lower means closer, so the winner neuron is the argmin of each row.
type SimilarityFunction = Callable[[Samples, Samples], npt.NDArray[Float]]
