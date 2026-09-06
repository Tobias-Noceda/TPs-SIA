from dataclasses import dataclass
from functools import cache
from typing import override

from tp2.interfaces.algorithms.fitness import FitnessCalculator

from tp2.models.individual_fitness import IndividualFitness

from tp2.algorithms.fitness.delta import DeltaEFitnessCalculator
from tp2.algorithms.fitness.overlap import OverlapFitnessCalculator

@dataclass(frozen=True, eq=False, unsafe_hash=False)
class MixedFitnessCalculator(FitnessCalculator):
    overlap: OverlapFitnessCalculator
    delta: DeltaEFitnessCalculator

    @override
    @staticmethod
    def for_image(path, lt = (0, 0), rb = None):
        overlap, size, color = OverlapFitnessCalculator.for_image(path, lt, rb)
        delta, _, _ = DeltaEFitnessCalculator.for_image(path, lt, rb)
        return (MixedFitnessCalculator(overlap, delta), size, color)

    @cache
    @override
    def calculate_fitness(self, individual):
        overlap = self.overlap.calculate_fitness(individual).score
        delta = self.delta.calculate_fitness(individual).score

        percent = 0.2
        fitness = overlap * percent + delta * (1 - percent)

        return IndividualFitness(individual, fitness)
