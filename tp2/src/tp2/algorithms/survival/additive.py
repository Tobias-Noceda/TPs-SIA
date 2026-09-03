from typing import Sequence

from tp2.models.individual_fitness import IndividualFitness

def additive_survival(N: Sequence[IndividualFitness], K: Sequence[IndividualFitness]):
    population = sorted([*N, *K], reverse=True)
    return population[:len(N)]
