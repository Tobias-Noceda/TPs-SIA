from typing import Callable, Tuple

from tp4.types.samples import Samples

type Transform = Callable[[Samples], Samples]

# Fits a normalization on the given samples and returns (normalize, denormalize).
# denormalize is None when the transformation can't be undone (e.g. unit length scaling).
type NormalizationFunction = Callable[[Samples], Tuple[Transform, Transform | None]]
