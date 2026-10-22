from typing import NamedTuple

from tp4.types.float import Float

class EpochResult(NamedTuple):
    """
    State of a self-organizing map at the end of an epoch.

    Attributes:
        epoch: Epoch number (0-based).
        learning_rate: η of the last iteration of the epoch.
        radius: R of the last iteration of the epoch.
        quantization_error: Mean distance between each sample and its winner's weights.
        topographic_error: Share of samples whose two closest neurons aren't adjacent on the map.
    """
    epoch: int
    learning_rate: Float
    radius: Float
    quantization_error: Float
    topographic_error: Float
