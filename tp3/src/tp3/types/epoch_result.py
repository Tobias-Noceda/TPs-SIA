from typing import Tuple

from tp3.types.float import Float

# Tuple of (epoch number, loss value, accuracy value, test_acuracy value) for each epoch of training.
type EpochResult = Tuple[int, Float, Float, Float]
