import random

from typing import Sequence, Tuple, Dict

from tp2.models.individual_fitness import IndividualFitness

from tp2.algorithms.selection._wheel import spin

# @link https://en.wikipedia.org/wiki/Stochastic_universal_sampling
def universal_selection(population: Sequence[IndividualFitness], data: Dict) -> Tuple[IndividualFitness, ...]:
    k: int = data["universal"]["select"]

    weights = [individual.score for individual in population]

    start = random.random()
    rolls = [(start + j) / k for j in range(k)]

    return spin(population, weights, rolls)
