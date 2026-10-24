from typing import NotRequired, TypedDict

from tp4.types.config.dataset import Dataset
from tp4.types.config.kohonen import KohonenDefinition
from tp4.types.config.method import Methods
from tp4.types.config.pca import PCADefinition
from tp4.types.schedule_function import ScheduleFunction

class Config(TypedDict):
    """
    Configuration of a run: the method and its blocks.

    Attributes:
        method: Method to run, it decides which of the blocks below are present.
        seed: Seed for every random draw (None = random).
        dataset: Dataset path, columns and normalization (the normalization isn't used by PCA).
        kohonen: Grid and functions of the map (kohonen).
        learning_rate: η(i) schedule (kohonen).
        radius: R(i) schedule (kohonen).
        epochs: Passes over the dataset, iterations = epochs · samples (kohonen).
        pca: Component selection and eigenvector signs (pca).
    """
    method: Methods
    seed: int | None
    dataset: Dataset
    kohonen: NotRequired[KohonenDefinition]
    learning_rate: NotRequired[ScheduleFunction]
    radius: NotRequired[ScheduleFunction]
    epochs: NotRequired[int]
    pca: NotRequired[PCADefinition]
