from tp4.types.float import Float
from tp4.types.schedule_function import ScheduleFunction

import numpy as np

# η(i) = 1/i, scaled by the initial value
def inverse_schedule(value: Float, final: Float | None = None) -> ScheduleFunction:
    """value / (i + 1) with 0-based i, never below final (default 0)."""
    floor = final if final is not None else np.float64(0)
    return lambda i, _: max(floor, value / (i + 1))
