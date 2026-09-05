from typing import Callable

from tp2.models.individual_fitness import IndividualFitness

type Selection = Callable[[list[IndividualFitness]], list[IndividualFitness]]