import random

from typing import Sequence, Tuple, Dict

from tp2.models.individual_fitness import IndividualFitness

from tp2.algorithms.selection._wheel import spin

# @link https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)#Rank_Selection
def ranking_selection(population: Sequence[IndividualFitness], data: Dict) -> Tuple[IndividualFitness, ...]:
    k: int = data["ranking"]["select"]

    size = len(population)
    ordered = sorted(population, reverse=True)

    weights = [(size - rank) / size for rank in range(1, size + 1)]
    rolls = [random.random() for _ in range(k)]

    return spin(ordered, weights, rolls)
