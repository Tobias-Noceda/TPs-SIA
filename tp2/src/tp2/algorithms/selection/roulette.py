import random

from typing import Sequence, Tuple, Dict

from tp2.models.individual_fitness import IndividualFitness

from tp2.algorithms.selection._wheel import spin

# @link https://en.wikipedia.org/wiki/Fitness_proportionate_selection
def roulette_selection(population: Sequence[IndividualFitness], data: Dict) -> Tuple[IndividualFitness, ...]:
    k: int = data["roulette"]["select"]

    weights = [individual.score for individual in population]
    rolls = [random.random() for _ in range(k)]

    return spin(population, weights, rolls)
