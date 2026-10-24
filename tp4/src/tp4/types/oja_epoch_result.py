from typing import NamedTuple

from tp4.types.float import Float

import numpy.typing as npt

class OjaEpochResult(NamedTuple):
    """
    State of the Oja neuron at the end of an epoch.

    Attributes:
        epoch: Epoch number (0-based).
        weights: Copy of w (n,) at the end of the epoch.
        norm: ‖w‖, tends to 1 when the rule converges.
    """
    epoch: int
    weights: npt.NDArray[Float]
    norm: Float
