from dataclasses import dataclass, field
from typing import Tuple, override

from tp3.interfaces.perceptron import Perceptron

from tp3.models.neuron import Neuron
from tp3.types.classification_option import ClassificationOption
from tp3.types.float import Float
from tp3.types.learning_rate_function import LearningRateFactory, LearningRateFunction
from tp3.types.loss_function import LossFunction
from tp3.types.perceptron_input import PerceptronInputs

import numpy as np
import numpy.typing as npt
from numpy.random import RandomState
from tp3.types.test_result import TestResult
from tqdm import trange

# Delta rule with mini-batch gradient descent (update rule given by the learning rate function)
# @link https://en.wikipedia.org/wiki/Delta_rule
@dataclass
class SimplePerceptron(Perceptron[Float]):
    neuron: Neuron

    learning_rate: LearningRateFactory
    loss_function: LossFunction

    input_size: int

    # Samples per weight update: 1 = online, len(inputs) = batch
    batch_size: int = 1

    # Training stops once the epoch error is below epsilon
    epsilon: Float = np.float64(1e-6)

    # Source of every random draw (weights and shuffling), seeded from the config for reproducibility
    random: RandomState = field(default_factory=RandomState, repr=False, hash=False)

    bias: Float = field(init=False, repr=False, hash=False)
    weights: npt.NDArray[Float] = field(init=False, repr=False, hash=False)

    # One stateful learning rate function per parameter
    weights_step: LearningRateFunction = field(init=False, repr=False, hash=False)
    bias_step: LearningRateFunction = field(init=False, repr=False, hash=False)

    def __post_init__(self):
        self.weights = self.random.uniform(-0.5, 0.5, size=self.input_size)
        self.bias = self.random.uniform(-0.5, 0.5, size=1)[0]
        self.weights_step = self.learning_rate()
        self.bias_step = self.learning_rate()

    @override
    def train(self, inputs, expected, epochs) -> Tuple[tuple[int, Float], ...]:
        inputs = self.__validate_parameters(inputs)
        expected = np.asarray(expected, dtype=np.float64)

        if len(inputs) != len(expected):
            raise ValueError("The number of inputs and expected outputs must be the same.")

        results = []
        # Error of the previous epoch, for the learning rate functions that adapt to it
        previous_error = np.float64(np.inf)
        for epoch in trange(epochs):
            # Outputs obtained while training, stored in dataset order to compute the epoch error
            outputs = np.empty_like(expected)

            order = self.random.permutation(len(inputs))
            for start in range(0, len(inputs), self.batch_size):
                batch = order[start:start + self.batch_size]

                h_u = inputs[batch] @ self.weights + self.bias

                output = self.neuron.activate(h_u)

                # ∂E/∂h = -(ζ - O) f'(h): the learning rate functions expect the real gradient and are subtracted
                delta = (output - expected[batch]) * self.neuron.derivate(h_u)

                outputs[batch] = output

                # Averaged over the batch so the effective learning rate does not depend on its size
                # The learning rate functions work on arrays: the scalar bias goes through as a 0-d one
                self.bias = np.float64(self.bias - self.bias_step(np.asarray(delta.mean()), previous_error))
                self.weights = self.weights - self.weights_step((inputs[batch].T @ delta) / len(batch), previous_error)

            error = self.loss_function(outputs, expected)
            results.append((epoch, error))
            previous_error = error

            if not np.isfinite(error):
                raise FloatingPointError(
                    f"Training diverged at epoch {epoch} (loss={error}). "
                    "Try a lower learning rate or normalize the inputs."
                )
            if error < self.epsilon:
                break

        return tuple(results)

    @override
    def test(self, inputs, expected, classification_function) -> TestResult:
        if len(inputs) != len(expected):
            raise ValueError("The number of inputs and expected outputs must be the same.")

        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        classifications = classification_function(tuple(self.predict(inputs)), tuple(expected))
        for i in range(len(classifications)):
            if classifications[i] == ClassificationOption.TRUE_POSITIVE:
                true_positives += 1
            elif classifications[i] == ClassificationOption.TRUE_NEGATIVE:
                true_negatives += 1
            elif classifications[i] == ClassificationOption.FALSE_POSITIVE:
                false_positives += 1
            elif classifications[i] == ClassificationOption.FALSE_NEGATIVE:
                false_negatives += 1

        return {
            "true_positive": true_positives,
            "true_negative": true_negatives,
            "false_positive": false_positives,
            "false_negative": false_negatives,
        }

    @override
    def predict(self, inputs) -> npt.NDArray[Float]:
        inputs = self.__validate_parameters(inputs)

        return self.neuron.activate(inputs @ self.weights + self.bias)

    def __validate_parameters(self, inputs: PerceptronInputs) -> PerceptronInputs:
        inputs = np.asarray(inputs, dtype=np.float64)
        if inputs.ndim != 2 or inputs.shape[1] != len(self.weights):
            raise ValueError(f"Inputs must be a (samples, {len(self.weights)}) matrix, got {inputs.shape}.")

        if self.batch_size < 1:
            raise ValueError("The batch size must be a positive integer.")

        return inputs
