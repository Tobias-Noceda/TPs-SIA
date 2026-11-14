from typing import Tuple

from tp5.types.float import Float

# (epoch number, loss, max wrong pixels over the patterns, patterns with at most goal_pixels wrong pixels)
# measured after the epoch, with its final weights.
type EpochResult = Tuple[int, Float, int, int]
