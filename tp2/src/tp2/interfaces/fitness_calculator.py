from abc import ABC, abstractmethod

from tp2.types.individual import Individual
    

class FitnessCalculator(ABC):
    """
    Interface for fitness calculators.
    """

    @abstractmethod
    def calculate_fitness(self, individual: Individual) -> float:
        """
        Calculates the fitness of an individual.
        """
        pass