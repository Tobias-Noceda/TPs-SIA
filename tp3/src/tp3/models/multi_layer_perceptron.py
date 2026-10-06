from dataclasses import dataclass, field
from typing import Callable, List, Tuple, override

from tp3.interfaces.optimizer import Optimizer
from tp3.interfaces.perceptron import Perceptron

from tp3.models.layer import Layer
from tp3.models.neuron import Neuron
from tp3.models.optimizers.gradient_descent import GradientDescent

from tp3.types.classification_option import ClassificationOption
from tp3.types.epoch_result import EpochResult
from tp3.types.loss_function import DerivativeLossFunction, LossFunction
from tp3.types.perceptron_input import PerceptronInputs
from tp3.types.float import Float

import numpy.typing as npt

import numpy as np
from tp3.types.test_result import TestResult
from tqdm import trange

# Backpropagation with mini-batch gradient descent
# @link https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Iterative_method
@dataclass
class MultiLayerPerceptron(Perceptron[npt.NDArray[Float]]):
    neurons: List[Tuple[Neuron, ...]]

    learning_rate: Float

    loss_function: LossFunction
    derivative_loss_function: DerivativeLossFunction

    input_size: int

    # Samples per weight update: 1 = online, len(inputs) = batch
    batch_size: int = 1

    # Builds one optimizer per layer (stateful optimizers keep per-layer state)
    optimizer: Callable[[], Optimizer] = GradientDescent

    layers: List[Layer] = field(init=False, repr=False, hash=False)

    epsilon = np.float64(1e-6)

    def __post_init__(self):
        super().__init__()

        self.layers = []
        prev_size = self.input_size
        for l in self.neurons:
            self.layers.append(Layer(prev_size, l, self.optimizer()))
            prev_size = len(l)

    @override
    def train(self, inputs, expected, epochs):
        inputs = self.__validate_parameters(inputs)
        expected = np.asarray(expected, dtype=np.float64)

        if len(inputs) != len(expected):
            raise ValueError("The number of inputs and expected outputs must be the same.")

        results: List[EpochResult] = []
        for epoch in trange(epochs):
            # Predictions made while training, stored in dataset order to compute the epoch loss
            predictions = np.empty_like(expected)

            order = np.random.permutation(len(inputs))
            for start in range(0, len(inputs), self.batch_size):
                batch = order[start:start + self.batch_size]
                expect = expected[batch]

                outputs = [inputs[batch]]
                activation_derivatives: List[npt.NDArray[Float]] = []

                for layer in self.layers:
                    activations, derivative = layer.fowards(outputs[-1])

                    outputs.append(activations)
                    activation_derivatives.append(derivative)

                prediction = outputs[-1]
                predictions[batch] = prediction

                loss_prime = self.derivative_loss_function(prediction, expect)

                deltas = [loss_prime * activation_derivatives[-1]]
                for layer, derivatives in zip(reversed(self.layers), reversed(activation_derivatives[:-1])):
                    delta_n = layer.delta(deltas[-1]) * derivatives
                    deltas.append(delta_n)
                deltas.reverse()

                for layer, data, delta in zip(self.layers, outputs, deltas):
                    layer.update(self.learning_rate, data, delta)

            loss = self.loss_function(predictions, expected)
            results.append((epoch, loss))

            if not np.isfinite(loss):
                raise FloatingPointError(
                    f"Training diverged at epoch {epoch} (loss={loss}). "
                    "Try a lower learning rate or normalize the inputs."
                )
            if loss < self.epsilon:
                break

        return tuple(results)

    @override
    def test(self, inputs, expected, classification_function):
        if len(inputs) != len(expected):
            raise ValueError("The number of inputs and expected outputs must be the same.")

        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        classifications = classification_function(tuple(self.predict(inputs)), tuple(expected))
        for classification in classifications:
            if classification == ClassificationOption.TRUE_POSITIVE:
                true_positives += 1
            elif classification == ClassificationOption.TRUE_NEGATIVE:
                true_negatives += 1
            elif classification == ClassificationOption.FALSE_POSITIVE:
                false_positives += 1
            elif classification == ClassificationOption.FALSE_NEGATIVE:
                false_negatives += 1

        return TestResult({
            "true_positive": true_positives,
            "true_negative": true_negatives,
            "false_positive": false_positives,
            "false_negative": false_negatives,
        })

    @override
    def predict(self, inputs):
        data = self.__validate_parameters(inputs)
        for layer in self.layers:
            data, _ = layer.fowards(data)
        return data

    def __validate_parameters(self, inputs: PerceptronInputs) -> PerceptronInputs:
        inputs = np.asarray(inputs, dtype=np.float64)
        if inputs.ndim != 2 or inputs.shape[1] != self.input_size:
            raise ValueError(f"Inputs must be a (samples, {self.input_size}) matrix, got {inputs.shape}.")

        if self.learning_rate <= 0:
            raise ValueError("The learning rate must be a positive value.")

        if self.batch_size < 1:
            raise ValueError("The batch size must be a positive integer.")

        return inputs
