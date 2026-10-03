from typing import Callable

from tp3.types.float import Float

type ActivationFunction = Callable[[Float], Float]

type DerivativeActivationFunction = Callable[[Float], Float]
