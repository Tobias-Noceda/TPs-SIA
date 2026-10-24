from dataclasses import dataclass, field
from typing import Tuple, override

from tp4.interfaces.unsupervised_network import UnsupervisedNetwork

from tp4.functions.hebbian.oja import oja

from tp4.types.float import Float
from tp4.types.initialization_function import InitializationFunction
from tp4.types.oja_epoch_result import OjaEpochResult
from tp4.types.samples import Samples

import numpy as np
import numpy.typing as npt
from numpy.random import RandomState
from tqdm import trange

# Single linear neuron trained with Oja's rule: w converges to the first principal component
# @link https://en.wikipedia.org/wiki/Oja%27s_rule
@dataclass
class Oja(UnsupervisedNetwork[OjaEpochResult]):
    """
    One linear neuron s = x · w whose weights learn the direction of largest variance.

    Samples must be centered (mean 0, e.g. z-score), otherwise w doesn't converge to the first
    eigenvector of the correlation matrix. η must be small for the number and scale of the
    features (η · ‖x‖² ≲ 1), otherwise the updates diverge.
    """
    # Constant η of every update
    learning_rate: Float

    initialization: InitializationFunction

    # Source of the initial weights, seeded from the config for reproducibility
    random: RandomState = field(default_factory=RandomState, repr=False, hash=False)

    # (n,), set by train
    weights: npt.NDArray[Float] = field(init=False, repr=False, hash=False)

    @override
    def train(self, inputs: Samples, epochs: int) -> Tuple[OjaEpochResult, ...]:
        # A new training may use other features, so they aren't checked against the current weights
        inputs = self.__validate_inputs(inputs, check_features=False)

        self.weights = self.initialization(self.random, inputs, 1)[0]

        results = []
        for epoch in trange(epochs):
            # Every sample, in the dataset's order
            for x in inputs:
                s = x @ self.weights
                self.weights += oja(self.weights, x, s, self.learning_rate)

            results.append(OjaEpochResult(epoch, self.weights.copy(), np.linalg.norm(self.weights)))

        if not np.all(np.isfinite(self.weights)):
            raise ValueError(f"The weights diverged with η = {self.learning_rate}: use a smaller learning rate.")

        return tuple(results)

    @override
    def predict(self, inputs: Samples) -> npt.NDArray[Float]:
        """Output s = x · w of each sample, the index: (P,)."""
        if not hasattr(self, "weights"):
            raise RuntimeError("The neuron must be trained first.")

        return self.__validate_inputs(inputs) @ self.weights

    def __validate_inputs(self, inputs: Samples, check_features: bool = True) -> Samples:
        inputs = np.asarray(inputs, dtype=np.float64)

        if inputs.ndim != 2:
            raise ValueError("Inputs must be a (samples, features) matrix.")
        if check_features and hasattr(self, "weights") and inputs.shape[1] != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} features, got {inputs.shape[1]}.")

        return inputs
