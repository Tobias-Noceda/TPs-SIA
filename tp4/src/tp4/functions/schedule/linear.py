from tp4.types.float import Float
from tp4.types.schedule_function import ScheduleFunction

import numpy as np

# Linear decay from the initial value to the final one (R(0) = grid size down to 1)
def linear_schedule(value: Float, final: Float | None = None) -> ScheduleFunction:
    """value at i = 0, final at the last iteration. Requires final."""
    if final is None:
        raise ValueError("A linear schedule requires a 'final' value.")

    return lambda i, total: value + (final - value) * i / max(total - 1, 1)
