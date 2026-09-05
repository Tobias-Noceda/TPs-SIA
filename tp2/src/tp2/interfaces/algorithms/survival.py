from typing import Callable, Sequence

from tp2.models.individual_fitness import IndividualFitness

type Survival = Callable[[Sequence[IndividualFitness], Sequence[IndividualFitness]], Sequence[IndividualFitness]]
