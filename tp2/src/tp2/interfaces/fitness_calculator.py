from abc import ABC, abstractmethod
from functools import cache

from tp2.models.individual_fitness import IndividualFitness
from tp2.types.coords import Coords
from tp2.types.individual import Individual
    

class FitnessCalculator(ABC):
    """
    Interface for fitness calculators.
    """

    @staticmethod
    @abstractmethod
    def for_image(path: str, lt: Coords = (0, 0), rb: Coords | None = None) -> tuple[FitnessCalculator, Coords]:
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
