from typing import NotRequired, TypedDict, Tuple

from tp3.types.config.columns import Columns
from tp3.types.augmentation_function import AugmentationFunction

# A dict containing the train, test, and validation dataset paths.
class Datasets(TypedDict):
    shape: NotRequired[Tuple[int, ...]]
    train: str
    test: str
    validation: str
    columns: Columns
    # Standardize inputs (z-score) with the training set statistics
    normalize: NotRequired[bool]
    augmentation_function: AugmentationFunction
