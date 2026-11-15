from dataclasses import dataclass, field
from typing import List, Tuple, override

from tp5.functions.pixel_error import pixel_errors
from tp5.interfaces.autoencoder import Autoencoder
from tp5.models.multi_layer_perceptron import MultiLayerPerceptron
from tp5.types.epoch_result import EpochResult
from tp5.types.float import Float
from tp5.types.loss_function import DerivativeLossFunction, LossFunction
from tp5.types.samples import Samples

import numpy as np
from numpy.random import RandomState
from tqdm import trange

# Two multi layer perceptrons, the decoder fed by the encoder, trained by backpropagation as a
# single network whose expected output is its own input.
# @link https://en.wikipedia.org/wiki/Autoencoder
@dataclass
class BasicAutoencoder(Autoencoder):
    encoder: MultiLayerPerceptron
    decoder: MultiLayerPerceptron

    loss_function: LossFunction
    derivative_loss_function: DerivativeLossFunction

    # Samples per weight update: 1 = online, len(inputs) = batch
    batch_size: int = 1

    # Training stops once the epoch loss is below epsilon
    epsilon: Float = np.float64(1e-6)

    # A pattern is learned when it has at most this many wrong pixels (None = no goal: the only
    # stop criteria are epsilon and the epochs limit)
    goal_pixels: int | None = None

    # Stop as soon as every pattern is learned (only with a goal)
    stop_at_goal: bool = True

    # Shuffles the samples of each epoch (the same generator initialized both networks)
    random: RandomState = field(default_factory=RandomState, repr=False, hash=False)

    def __post_init__(self) -> None:
        if self.encoder.output_size != self.decoder.input_size:
            raise ValueError(
                f"The encoder outputs {self.encoder.output_size} values but the decoder expects {self.decoder.input_size}."
            )
        if self.decoder.output_size != self.encoder.input_size:
            raise ValueError(
                f"The decoder outputs {self.decoder.output_size} values but the encoder expects {self.encoder.input_size}."
            )
        if self.batch_size < 1:
            raise ValueError("The batch size must be a positive integer.")

    @property
    def latent_size(self) -> int:
        return self.encoder.output_size

    @override
    def encode(self, inputs: Samples) -> Samples:
        return self.encoder.predict(self.__validate(inputs, self.encoder.input_size))

    @override
    def decode(self, latent: Samples) -> Samples:
        return self.decoder.predict(self.__validate(latent, self.decoder.input_size))

    @override
    def train(self, inputs: Samples, expected: Samples, epochs: int) -> Tuple[EpochResult, ...]:
        inputs = self.__validate(inputs, self.encoder.input_size)
        expected = self.__validate(expected, self.decoder.output_size)
        if len(inputs) != len(expected):
            raise ValueError("The number of inputs and expected outputs must be the same.")

        results: List[EpochResult] = []
        # Error of the previous epoch, for the learning rate functions that adapt to it
        previous_loss = np.float64(np.inf)
        for epoch in trange(epochs, mininterval=0.5):
            order = self.random.permutation(len(inputs))
            for start in range(0, len(inputs), self.batch_size):
                batch = order[start:start + self.batch_size]

                output = self.decoder.forward(self.encoder.forward(inputs[batch]))
                gradient = self.derivative_loss_function(output, expected[batch])
                self.encoder.backward(self.decoder.backward(gradient))

                self.decoder.step(previous_loss)
                self.encoder.step(previous_loss)

            # Measured with the weights at the end of the epoch (32 patterns: cheap), so the stop
            # criterion refers to the model that is returned
            reconstructions = self.reconstruct(inputs)
            loss = self.loss_function(reconstructions, expected)
            errors = pixel_errors(reconstructions, expected)
            # Without a goal, a pattern counts as learned only when reconstructed exactly
            allowed = self.goal_pixels if self.goal_pixels is not None else 0
            results.append((epoch, loss, int(errors.max()), int(np.sum(errors <= allowed))))
            previous_loss = loss

            if not np.isfinite(loss):
                raise FloatingPointError(
                    f"Training diverged at epoch {epoch} (loss={loss}). Try a lower learning rate."
                )
            if loss < self.epsilon or (self.stop_at_goal and self.goal_pixels is not None and errors.max() <= self.goal_pixels):
                break

        return tuple(results)

    @staticmethod
    def __validate(data: Samples, size: int) -> Samples:
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != size:
            raise ValueError(f"Expected a (samples, {size}) matrix, got {data.shape}.")
        return data
