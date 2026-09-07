from dataclasses import dataclass
from functools import cache
from typing import override

from tp2.interfaces.algorithms.fitness import ImageFitnessCalculator

from tp2.models.image_parser import ImageParser
from tp2.models.individual_fitness import IndividualFitness

import numpy as np
from skimage.metrics import structural_similarity

@dataclass(frozen=True, eq=False, unsafe_hash=False)
class SSIMFitnessCalculator(ImageFitnessCalculator):
    reference: np.ndarray

    width: int
    height: int
    window_size: int

    background: tuple[int, ...]

    @override
    @staticmethod
    def for_image(path, lt = (0, 0), rb = None):
        reference, width, height, color = ImageParser._open_image(path, lt, rb)

        min_dim = min(reference.shape[0], reference.shape[1])
        window_size = min(7, round(min_dim / 2 + .01) * 2 - 1) # Make it odd

        return (SSIMFitnessCalculator(reference, width, height, window_size, color), (width, height), color)

    @override
    @cache
    def calculate_fitness(self, individual):
        rendered = ImageParser._get_individual_render(
            individual,
            self.width,
            self.height,
            self.background
        )

        fitness: float = structural_similarity(
            self.reference, rendered,
            data_range=rendered.max() - rendered.min(),
            channel_axis=-1,
            win_size=self.window_size
        ) # pyright: ignore[reportAssignmentType]

        return IndividualFitness(individual, fitness)
