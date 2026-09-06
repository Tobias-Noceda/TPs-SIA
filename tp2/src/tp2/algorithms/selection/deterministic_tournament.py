import random

from typing import Sequence, Tuple, Dict

from tp2.models.individual_fitness import IndividualFitness

# @link https://en.wikipedia.org/wiki/Tournament_selection
def deterministic_tournament_selection(population: Sequence[IndividualFitness], data: Dict) -> Tuple[IndividualFitness, ...]:
    config = data["deterministic_tournament"]

    k: int = config["select"]
    m: int = min(config["m"], len(population))

    return tuple(
        max(random.sample(population, m))
        for _ in range(k)
    )
