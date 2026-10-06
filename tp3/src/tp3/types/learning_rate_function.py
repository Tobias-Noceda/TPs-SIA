from typing import Callable

from tp3.types.float import Float

import numpy.typing as npt

# Update rule for ONE parameter array, called once per batch:
#   (gradient: ∂E/∂θ averaged over the batch, error: error of the previous epoch, inf on the first one)
#   -> step to subtract (θ -= step), same shape as the gradient
# Stateful (velocity, moments, current η) through its closure.
type LearningRateFunction = Callable[[npt.NDArray[Float], Float], npt.NDArray[Float]]

# Builds a fresh LearningRateFunction (empty state) for each parameter array (W and b of each layer):
# sharing one across parameters would mix their histories and shapes.
type LearningRateFactory = Callable[[], LearningRateFunction]
