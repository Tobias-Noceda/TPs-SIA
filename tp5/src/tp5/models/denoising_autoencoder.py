from dataclasses import dataclass, field
from typing import Tuple, override

from tp5.functions.noise.flip import flip_noise
from tp5.models.basic_autoencoder import BasicAutoencoder
from tp5.types.noise_function import NoiseFunction
from tp5.types.samples import Samples

import numpy as np

# Denoising autoencoder: trained with corrupted inputs X̃ and the clean X as expected output, so it
# learns to map noisy patterns back to the originals.
# @link https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf
@dataclass
class DenoisingAutoencoder(BasicAutoencoder):
    # Corruption of the inputs: probability per pixel (flip, salt_pepper) or σ (gaussian)
    noise: NoiseFunction = field(default=flip_noise, kw_only=True)
    noise_level: float = field(default=0.1, kw_only=True)

    # Noisy versions of every pattern per epoch (data augmentation)
    copies: int = field(default=1, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.copies < 1:
            raise ValueError("The number of noisy copies must be a positive integer.")
        if not 0 <= self.noise_level <= 1:
            raise ValueError("The noise level must be in [0, 1].")

    def corrupt(self, samples: Samples) -> Samples:
        """`copies` noisy versions of the samples, stacked (rows i, i + N, i + 2N... are copies of sample i)."""
        return self.noise(self.random, np.tile(samples, (self.copies, 1)), self.noise_level)

    @override
    def _epoch_data(self, inputs: Samples, expected: Samples) -> Tuple[Samples, Samples]:
        # New corruption every epoch: the network never sees the same noisy pattern twice
        return self.corrupt(inputs), np.tile(expected, (self.copies, 1))

    @override
    def _evaluation_data(self, inputs: Samples, expected: Samples) -> Tuple[Samples, Samples]:
        # Fixed noisy set: the epoch metrics (loss, wrong pixels, learned patterns) and the stop
        # criteria measure denoising (output vs clean) on the same corruption every epoch
        return self.corrupt(inputs), np.tile(expected, (self.copies, 1))
