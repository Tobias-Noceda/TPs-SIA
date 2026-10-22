from typing import TypedDict

from tp4.types.config.dataset import Dataset
from tp4.types.config.kohonen import KohonenDefinition
from tp4.types.schedule_function import ScheduleFunction

class KohonenConfig(TypedDict):
    """
    Configuration for the Kohonen network.

    Attributes:
        kohonen: Grid and functions of the map.
        learning_rate: η(i) schedule.
        radius: R(i) schedule.
        epochs: Passes over the dataset (iterations = epochs · samples).
        seed: Seed for weight initialization and shuffling (None = random).
        dataset: Dataset path, columns and normalization.
    """
    kohonen: KohonenDefinition
    learning_rate: ScheduleFunction
    radius: ScheduleFunction
    epochs: int
    seed: int | None
    dataset: Dataset
