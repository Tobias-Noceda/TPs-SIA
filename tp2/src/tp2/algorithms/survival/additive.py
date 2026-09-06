from typing import Sequence, Tuple

from tp2.models.individual_fitness import IndividualFitness

def additive_survival(N: Sequence[IndividualFitness], K: Sequence[IndividualFitness]) -> Tuple[IndividualFitness, ...]:
    population = [*N, *K]
    population = sorted(population, reverse=True)
    return tuple(population[:len(N)])
