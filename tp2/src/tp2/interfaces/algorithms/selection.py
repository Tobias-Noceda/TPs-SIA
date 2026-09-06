from typing import Callable, Dict, Sequence, Tuple

from tp2.models.individual_fitness import IndividualFitness

type Selection = Callable[[Sequence[IndividualFitness], Dict], Tuple[IndividualFitness, ...]]
