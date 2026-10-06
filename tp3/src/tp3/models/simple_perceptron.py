from dataclasses import dataclass, field
from typing import Tuple, List, override

from tp3.interfaces.perceptron import Perceptron

from tp3.models.neuron import Neuron
from tp3.types.classification_option import ClassificationOption
from tp3.types.float import Float
from tp3.types.loss_function import LossFunction
from tp3.types.perceptron_input import PerceptronInput

import numpy as np
import numpy.typing as npt
from tp3.types.test_result import TestResult
from tqdm import trange

@dataclass
class SimplePerceptron(Perceptron[Float], Neuron):

    learning_rate: Float
    loss_function: LossFunction
    
    input_size: int
    
    bias: Float = field(init=False, repr=False, hash=False)
    weights: npt.NDArray[Float] = field(init=False, repr=False, hash=False)

    random: np.random.RandomState
    epsilon: Float

    def __post_init__(self):
        print(f"Initializing SimplePerceptron with input_size={self.input_size}, learning_rate={self.learning_rate}, epsilon={self.epsilon}")
        self.weights = self.random.uniform(-0.5, 0.5, size=self.input_size)
        self.bias = self.random.uniform(-0.5, 0.5, size=1)[0]

    @override
    def train(self, inputs, expected, epochs) -> Tuple[tuple[int, Float], ...]:        
        if len(inputs) != len(expected):
            raise ValueError("The number of inputs and expected outputs must be the same.")

        self.__validate_parameters(inputs)
    
        results = []
        for epoch in trange(epochs):
            outputs: List[Float] = []
            for u in range(len(inputs)):
                h_u = np.dot(self.weights, inputs[u]) + self.bias
                
                output = self.activate(h_u)

                delta = self.learning_rate * (expected[u] - output) * self.derivate(h_u)

                outputs.append(output)

                self.bias = self.bias + delta
                self.weights = self.weights + delta * inputs[u]

            error = self.loss_function(np.array(outputs), np.array(expected))
            results.append((epoch, error))

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
        self.__validate_parameters(inputs)
        
        if len(inputs) != len(expected):
            raise ValueError("The number of inputs and expected outputs must be the same.")

        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        classifications = classification_function(self.predict(inputs), tuple(expected))
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
    def predict(self, inputs) -> Tuple[Float, ...]:
        self.__validate_parameters(inputs)

        results = []
        for u in range(len(inputs)):
            h_u = np.dot(self.weights, inputs[u]) + self.bias
            results.append(self.activate(h_u))

        return tuple(results)

    def __validate_parameters(self, inputs: Tuple[PerceptronInput, ...]):
        input_dim = len(inputs[0])
        for i in range(len(inputs)):
            if len(inputs[i]) != input_dim:
                raise ValueError("The number of input features must match the number of weights.")

        if self.weights is not None and input_dim != len(self.weights):
            raise ValueError("Input dimension does not match initialized weights.")

        if self.learning_rate <= 0:
            raise ValueError("The learning rate must be a positive value.")