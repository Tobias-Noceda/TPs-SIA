from abc import ABC, abstractmethod
from typing import Tuple

from tp5.types.epoch_result import EpochResult
from tp5.types.samples import Samples

# Encoder f: X → Z (latent space) and decoder g: Z → X', trained so that X' ≈ X.
class Autoencoder(ABC):

    @abstractmethod
    def encode(self, inputs: Samples) -> Samples:
        pass

    @abstractmethod
    def decode(self, latent: Samples) -> Samples:
        pass

    def reconstruct(self, inputs: Samples) -> Samples:
        return self.decode(self.encode(inputs))

    @abstractmethod
    def train(self, inputs: Samples, expected: Samples, epochs: int) -> Tuple[EpochResult, ...]:
        """
        Train with `inputs` as the network input and `expected` as the target output (the same
        matrix for a basic autoencoder).
        """
        pass
