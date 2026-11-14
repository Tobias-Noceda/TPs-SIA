from dataclasses import dataclass

from tp5.types.activation_function import ActivationFunction, DerivativeActivationFunction
from tp5.types.float import Float

import numpy.typing as npt

@dataclass(frozen=True)
class Neuron:
    activation_function: ActivationFunction
    derivative_activation_function: DerivativeActivationFunction

    def activate(self, x: npt.NDArray[Float]):
        return self.activation_function(x)

    def derivate(self, x: npt.NDArray[Float]):
        return self.derivative_activation_function(x)
