from typing import Sequence, Tuple

from tp2.models.individual_fitness import IndividualFitness

def exclusive_survival(N: Sequence[IndividualFitness], K: Sequence[IndividualFitness]) -> Tuple[IndividualFitness, ...]:
    Ks = sorted(K, reverse=True)

    if len(K) > len(N):
        return tuple(Ks[:len(N)])

    Ns = sorted(N, reverse=True)

    population = Ks

    fill = len(N) - len(K)
    population += Ns[:fill]

    return tuple(sorted(population, reverse=True))
