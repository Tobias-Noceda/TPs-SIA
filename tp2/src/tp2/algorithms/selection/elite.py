from math import ceil
from typing import Sequence, Tuple, Dict

from tp2.models.individual_fitness import IndividualFitness

# @link https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)
def elite_selection(population: Sequence[IndividualFitness], data: Dict) -> Tuple[IndividualFitness, ...]:
    k: int = data["elite"]["select"]

    size = len(population)
    ordered = sorted(population, reverse=True)

    return tuple(
        ordered[i]
        for i in range(size)
        for _ in range(max(ceil((k - i) / size), 0))
    )
