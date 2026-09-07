from abc import ABC, abstractmethod
from functools import cache
from typing import override

from tp2.models.individual_fitness import IndividualFitness
from tp2.types.coords import Coords
from tp2.types.individual import Individual

class ImageFitnessCalculator(ABC):
    """
    Interface for fitness calculators.
    """

    @staticmethod
    @abstractmethod
    def for_image(path: str, lt: Coords = (0, 0), rb: Coords | None = None) -> tuple[ImageFitnessCalculator, Coords, tuple[int, ...]]:
        """
        Left-Top coords of the image region to check
        Right-Bottom coords of the image region to check
        """
        pass

    @cache
    @abstractmethod
    def calculate_fitness(self, individual: Individual) -> IndividualFitness:
        """
        Calculates the fitness of an individual.
        """
        pass

    @override
    def __eq__(self, value: object) -> bool:
        return super().__eq__(value)

    @override
    def __hash__(self) -> int:
        return super().__hash__()
