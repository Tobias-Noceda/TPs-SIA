from typing import Callable, Tuple

import numpy as np

type ActivationFunction = Callable[[Tuple[np.float32, ...]], np.float32]
