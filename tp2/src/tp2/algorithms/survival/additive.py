from typing import Sequence

from tp2.models.individual_fitness import IndividualFitness

def additive_selection(old: Sequence[IndividualFitness], new: Sequence[IndividualFitness]):
    population = sorted([*old, *new], reverse=True)
    return population[:len(old)]
