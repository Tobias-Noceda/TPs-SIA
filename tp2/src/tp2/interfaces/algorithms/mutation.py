from typing import Callable

from tp2.models.individual_fitness import IndividualFitness
from tp2.types.individual import Individual

# Represents a mutation function that takes an IndividualFitness and a float (mutation_rate 0 < mutation_rate <= 1) as input and returns a mutated Individual.
type Mutation = Callable[[IndividualFitness, float], Individual]