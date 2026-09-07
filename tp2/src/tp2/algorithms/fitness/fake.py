from dataclasses import dataclass
from functools import cache
from typing import override

from tp2.interfaces.algorithms.fitness import ImageFitnessCalculator
from tp2.models.individual_fitness import IndividualFitness

@dataclass(frozen=True)
class FakeFitnessCalculator(ImageFitnessCalculator):
    @override
    @staticmethod
    def for_image(path, lt = (0, 0), rb = None):
        return (FakeFitnessCalculator(), (500, 500), tuple())  # Placeholder values for width and height

    @cache
    @override
    def calculate_fitness(self, individual):
        """
        Calculates the fitness of an image individual based on its similarity to the target image.
        The fitness is calculated as the inverse of the average pixel difference between the individual's image and the target image.
        A higher fitness score indicates a closer match to the target image.
        """
        return IndividualFitness(individual, 0.5)  # Placeholder
