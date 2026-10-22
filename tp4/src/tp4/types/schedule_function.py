from typing import Callable

from tp4.types.float import Float

# Value of a decaying hyperparameter (learning rate η, radius R) at iteration i out of the total.
type ScheduleFunction = Callable[[int, int], Float]
