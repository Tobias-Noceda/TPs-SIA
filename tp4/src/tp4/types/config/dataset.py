from typing import List, NotRequired, TypedDict

from tp4.types.normalization_function import NormalizationFunction

class Columns(TypedDict):
    # Numeric columns that make up each sample
    input: List[str]
    # Column that names each sample (e.g. the creator), only used to label results
    label: NotRequired[str]

# A dict containing the dataset path, its columns and how to normalize it.
class Dataset(TypedDict):
    path: str
    columns: Columns
    normalize: NormalizationFunction
