from typing import Callable, Tuple

from tp2.types.individual import Individual

type Breeding = Callable[[Tuple[Individual, Individual]], Tuple[Individual, Individual]]