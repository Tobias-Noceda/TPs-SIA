import random

from typing import Sequence, Tuple, Dict

from tp2.models.individual_fitness import IndividualFitness

# @link https://en.wikipedia.org/wiki/Tournament_selection
def probabilistic_tournament_selection(population: Sequence[IndividualFitness], data: Dict) -> Tuple[IndividualFitness, ...]:
    config = data["probabilistic_tournament"]

    k: int = config["select"]
    threshold: float = config["threshold"]

    return tuple(
        _duel(population, threshold)
        for _ in range(k)
    )

def _duel(population: Sequence[IndividualFitness], threshold: float) -> IndividualFitness:
    if len(population) < 2:
        return population[0]

    a, b = random.sample(population, 2)
    best, worst = (a, b) if a > b else (b, a)

    return best if random.random() < threshold else worst
