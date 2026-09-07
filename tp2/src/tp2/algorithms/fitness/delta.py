from dataclasses import dataclass
from functools import cache
from typing import override

from tp2.interfaces.algorithms.fitness import ImageFitnessCalculator

from tp2.models.image_parser import ImageParser
from tp2.models.individual_fitness import IndividualFitness

import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000

@dataclass(frozen=True, eq=False, unsafe_hash=False)
class DeltaEFitnessCalculator(ImageFitnessCalculator):
    reference_lab: np.ndarray

    width: int
    height: int

    background: tuple[int, ...]

    @override
    @staticmethod
    def for_image(path, lt = (0, 0), rb = None):
        reference, width, height, color = ImageParser._open_image(path, lt, rb)
        return (DeltaEFitnessCalculator(rgb2lab(reference), width, height, color), (width, height), color)

    @cache
    @override
    def calculate_fitness(self, individual):
        """
        Calculates the fitness of an individual.
        """
        rendered = ImageParser._get_individual_render(
            individual,
            self.width,
            self.height,
            self.background
        ).astype(np.float32) / 255.0

        rendered_lab = rgb2lab(rendered)

        de: np.ndarray = deltaE_ciede2000(self.reference_lab, rendered_lab)

        # print("Mean ΔE:", np.mean(de))
        # print("Median ΔE:", np.median(de))
        # print("95th percentile ΔE:", np.percentile(de, 95))
        # print("Max ΔE:", np.max(de))

        return IndividualFitness(individual, 1 / (1 + np.mean(de)))
