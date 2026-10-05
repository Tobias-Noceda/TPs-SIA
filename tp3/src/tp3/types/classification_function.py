from typing import Callable, Tuple

from tp3.types.classification_option import ClassificationOption

type ClassificationFunction[OT] = Callable[[Tuple[OT, ...], Tuple[OT, ...]], Tuple[ClassificationOption, ...]]
