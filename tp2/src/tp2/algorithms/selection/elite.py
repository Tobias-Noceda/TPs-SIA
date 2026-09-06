from typing import Sequence, Tuple, Dict

from tp2.models.individual_fitness import IndividualFitness

def elite_selection(population: Sequence[IndividualFitness], data: Dict) -> Tuple[IndividualFitness, ...]:
    population = sorted(population, reverse=True)
    return tuple(population[:data["elite"]["select"]])
