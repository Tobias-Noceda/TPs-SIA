from dataclasses import dataclass, field
from typing import Tuple, override

from tp4.interfaces.unsupervised_network import UnsupervisedNetwork

from tp4.functions.metrics.quantization_error import quantization_error
from tp4.functions.metrics.topographic_error import topographic_error

from tp4.types.epoch_result import EpochResult
from tp4.types.float import Float
from tp4.types.grid import GridDistances, GridPositions
from tp4.types.initialization_function import InitializationFunction
from tp4.types.neighborhood_function import NeighborhoodFunction
from tp4.types.samples import Samples
from tp4.types.schedule_function import ScheduleFunction
from tp4.types.similarity_function import SimilarityFunction
from tp4.types.topology import Topology

import numpy as np
import numpy.typing as npt
from numpy.random import RandomState
from tqdm import trange

# Self-organizing map trained with the Kohonen rule
# @link https://en.wikipedia.org/wiki/Self-organizing_map
@dataclass
class Kohonen(UnsupervisedNetwork[EpochResult]):
    """
    k x k map of neurons, each with a weight vector of the same dimension as the samples.

    Samples should be standardized: the winner is picked by distance, so features
    with larger scales would dominate it.
    """
    grid_size: int

    topology: Topology
    similarity: SimilarityFunction
    neighborhood: NeighborhoodFunction

    # η(i) and R(i), evaluated with the global iteration i out of epochs · samples
    learning_rate: ScheduleFunction
    radius: ScheduleFunction

    initialization: InitializationFunction

    # Source of every random draw (initial weights and sample order), seeded from the config for reproducibility
    random: RandomState = field(default_factory=RandomState, repr=False, hash=False)

    positions: GridPositions = field(init=False, repr=False, hash=False)
    grid_distances: GridDistances = field(init=False, repr=False, hash=False)

    # (neurons, n), set by train
    weights: npt.NDArray[Float] = field(init=False, repr=False, hash=False)

    def __post_init__(self):
        if self.grid_size < 2:
            raise ValueError("The grid size must be at least 2.")

        self.positions = self.topology.positions(self.grid_size)
        self.grid_distances = np.linalg.norm(self.positions[:, None, :] - self.positions[None, :, :], axis=2)

    @property
    def neurons(self) -> int:
        return self.grid_size * self.grid_size

    @override
    def train(self, inputs: Samples, epochs: int) -> Tuple[EpochResult, ...]:
        # A new training may use other features, so they aren't checked against the current weights
        inputs = self.__validate_inputs(inputs, check_features=False)

        self.weights = self.initialization(self.random, inputs, self.neurons)

        total = epochs * len(inputs)
        eta = radius = np.float64(0)

        results = []
        for epoch in trange(epochs):
            # One sample per iteration, in a new random order every epoch
            for step, sample in enumerate(self.random.permutation(len(inputs))):
                i = epoch * len(inputs) + step
                eta = self.learning_rate(i, total)
                radius = self.radius(i, total)

                x = inputs[sample]
                winner = np.argmin(self.similarity(self.weights, x[None, :])[0])

                strength = self.neighborhood(self.grid_distances[winner], radius)
                self.weights += eta * strength[:, None] * (x - self.weights)

            scores = self.scores(inputs)
            results.append(EpochResult(
                epoch,
                eta,
                radius,
                quantization_error(inputs, self.weights, np.argmin(scores, axis=1)),
                topographic_error(scores, self.grid_distances),
            ))

        return tuple(results)

    @override
    def predict(self, inputs: Samples) -> npt.NDArray[np.intp]:
        """Winner neuron index of each sample: (P,)."""
        return np.argmin(self.scores(self.__validate_inputs(inputs)), axis=1)

    def scores(self, inputs: Samples) -> npt.NDArray[Float]:
        """Dissimilarity between each sample and each neuron: (P, neurons)."""
        if not hasattr(self, "weights"):
            raise RuntimeError("The map must be trained first.")

        return self.similarity(self.weights, inputs)

    def coordinates(self, neurons: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
        """(row, column) of each neuron index on the k × k grid: (len(neurons), 2)."""
        return np.column_stack(np.divmod(neurons, self.grid_size))

    def __validate_inputs(self, inputs: Samples, check_features: bool = True) -> Samples:
        inputs = np.asarray(inputs, dtype=np.float64)

        if inputs.ndim != 2:
            raise ValueError("Inputs must be a (samples, features) matrix.")
        if check_features and hasattr(self, "weights") and inputs.shape[1] != self.weights.shape[1]:
            raise ValueError(f"Expected {self.weights.shape[1]} features, got {inputs.shape[1]}.")

        return inputs
