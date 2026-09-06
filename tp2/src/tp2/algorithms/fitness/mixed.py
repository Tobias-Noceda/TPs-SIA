from dataclasses import dataclass
from functools import cache
from typing import override

from tp2.interfaces.algorithms.fitness import FitnessCalculator

from tp2.models.individual_fitness import IndividualFitness

from tp2.algorithms.fitness.delta import DeltaEFitnessCalculator
from tp2.algorithms.fitness.ssim import SSIMFitnessCalculator

@dataclass(frozen=True, eq=False, unsafe_hash=False)
class MixedFitnessCalculator(FitnessCalculator):
    ssim: SSIMFitnessCalculator
    delta: DeltaEFitnessCalculator

    @override
    @staticmethod
    def for_image(path, lt = (0, 0), rb = None):
        ssim, size, color = SSIMFitnessCalculator.for_image(path, lt, rb)
        delta, _, _ = DeltaEFitnessCalculator.for_image(path, lt, rb)
        return (MixedFitnessCalculator(ssim, delta), size, color)

    @cache
    @override
    def calculate_fitness(self, individual):
        ssim = self.ssim.calculate_fitness(individual).score
        delta = self.delta.calculate_fitness(individual).score

        percent = 0.2
        fitness = ssim * percent + delta * (1 - percent)

        return IndividualFitness(individual, fitness)
