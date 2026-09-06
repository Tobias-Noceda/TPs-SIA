from dataclasses import dataclass
from functools import cache
from typing import override

from tp2.interfaces.algorithms.fitness import FitnessCalculator
from tp2.types.individual import Individual

from tp2.models.individual_fitness import IndividualFitness

import numpy as np
from PIL import Image, ImageDraw

@dataclass(frozen=True, eq=False, unsafe_hash=False)
class OverlapFitnessCalculator(FitnessCalculator):
    reference: np.ndarray

    width: int
    height: int

    @override
    @staticmethod
    def for_image(path, lt = (0, 0), rb = None):
        reference, width, height, color = FitnessCalculator._open_image(path, lt, rb)
        return (OverlapFitnessCalculator(reference, width, height), (width, height), color)

    @cache
    @override
    def calculate_fitness(self, individual):
        overlaps = OverlapFitnessCalculator._get_individual_overlap(
            individual,
            self.width,
            self.height
        )

        fitness = 1 - np.max(overlaps) / len(individual)

        return IndividualFitness(individual, fitness)

    @staticmethod
    def _get_individual_overlap(individual: Individual, w: int, h: int):
        coverage = np.zeros((h, w), dtype=np.uint16)

        for polygon in individual:
            mask = Image.new("1", (w, h), 0)
            draw = ImageDraw.Draw(mask)

            draw.polygon(polygon.vertices, fill=1)

            coverage += np.asarray(mask, dtype=np.uint16)

        return coverage
