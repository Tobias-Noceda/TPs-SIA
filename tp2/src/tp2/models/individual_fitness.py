from dataclasses import dataclass
from functools import total_ordering

from tp2.interfaces.fitness_calculator import FitnessCalculator
from tp2.types.individual import Individual

@dataclass(eq=False, frozen=True)
@total_ordering
class IndividualFitness:
    individual: Individual
    score: float
    
    @staticmethod
    def calculate(individual: Individual, calculator: FitnessCalculator) -> IndividualFitness:
        """
        Represents the fitness of an individual in the genetic algorithm.
        """
        return IndividualFitness(individual, calculator.calculate_fitness(individual))

    def __lt__(self, other: IndividualFitness) -> bool:
        """
        Compares two IndividualFitness objects based on their score.
        """
        return self.score < other.score

    def __eq__(self, other) -> bool:
        """
        Compares two IndividualFitness objects based on their score.
        """
        if not isinstance(other, IndividualFitness) or other is None:
            return False
        
        return self.score == other.score