from typing import NotRequired, TypedDict

from tp3.types.config.columns import Columns

# A dict containing the train, test, and validation dataset paths.
class Datasets(TypedDict):
    train: str
    test: str
    validation: str
    columns: Columns
    # Standardize inputs (z-score) with the training set statistics
    normalize: NotRequired[bool]