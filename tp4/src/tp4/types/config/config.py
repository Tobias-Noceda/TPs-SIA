from typing import NotRequired, TypedDict

from tp4.types.config.dataset import Dataset
from tp4.types.config.hopfield import HopfieldDefinition
from tp4.types.config.kohonen import KohonenDefinition
from tp4.types.config.method import Methods
from tp4.types.config.oja import OjaDefinition
from tp4.types.config.pca import PCADefinition
from tp4.types.schedule_function import ScheduleFunction

class PCAConfig(TypedDict):
    """
    Configuration of a PCA run.

    Attributes:
        dataset: Dataset path, columns.
        pca: Component selection and eigenvector signs (pca).
    """
    dataset: Dataset
    pca: PCADefinition

class NeuralNetworkConfig(PCAConfig):
    """
    Configuration of a Neural Network run: the method and its blocks.

    Attributes:
        method: Method to run, it decides which network architecture is used.
        seed: Seed for every random draw (None = random).
        dataset: Dataset path, columns and normalization.
        kohonen: Grid and functions of the map (kohonen).
        learning_rate: η(i) schedule.
        radius: R(i) schedule.
        epochs: Passes over the dataset, iterations = epochs · samples.
    """
    method: Methods
    seed: int | None
    dataset: Dataset
    kohonen: NotRequired[KohonenDefinition]
    learning_rate: ScheduleFunction
    radius: ScheduleFunction
    epochs: int

class OjaConfig(TypedDict):
    """
    Configuration of an Oja run.

    Attributes:
        method: Always oja.
        seed: Seed for the initial weights (None = random).
        dataset: Dataset path, columns and normalization (z_score).
        oja: Constant learning rate and initial weights (oja).
        epochs: Passes over the dataset, in its order.
    """
    method: Methods
    seed: int | None
    dataset: Dataset
    oja: OjaDefinition
    epochs: int

class HopfieldConfig(TypedDict):
    """
    Configuration of a Hopfield run.

    Attributes:
        method: Always hopfield.
        seed: Seed for the noise and the asynchronous update order (None = random).
        hopfield: Patterns, update rule and queries (hopfield).
    """
    method: Methods
    seed: int | None
    hopfield: HopfieldDefinition

type Config = PCAConfig | NeuralNetworkConfig | OjaConfig | HopfieldConfig
