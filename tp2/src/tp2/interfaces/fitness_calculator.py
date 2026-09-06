from abc import ABC, abstractmethod
from functools import cache

from tp2.types.individual import Individual
    

class FitnessCalculator(ABC):
    """
    Interface for fitness calculators.
    """

    @cache
    @abstractmethod
    def calculate_fitness(self, individual: Individual) -> float:
        """
        Calculates the fitness of an individual.
        """
        pass
