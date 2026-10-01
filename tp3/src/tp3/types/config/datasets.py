from typing import TypedDict


# A dict containing the train, test, and validation dataset paths.
class DatasetPaths(TypedDict):
    train: str
    test: str
    validation: str