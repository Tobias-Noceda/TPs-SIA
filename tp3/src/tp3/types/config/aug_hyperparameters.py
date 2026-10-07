from typing import TypedDict, Tuple

class AugmentationFunctionHyperparameters(TypedDict, total=False):
    max_shift: Tuple[int, ...]  # random shift
