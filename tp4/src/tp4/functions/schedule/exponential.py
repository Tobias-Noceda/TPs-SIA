from tp4.types.float import Float
from tp4.types.schedule_function import ScheduleFunction

# Exponential (geometric) decay from the initial value to the final one
def exponential_schedule(value: Float, final: Float | None = None) -> ScheduleFunction:
    """value · (final / value)^(i / (total - 1)): value at i = 0, final at the last iteration. Requires 0 < final, value."""
    if final is None:
        raise ValueError("An exponential schedule requires a 'final' value.")
    if value <= 0 or final <= 0:
        raise ValueError("An exponential schedule requires positive 'value' and 'final'.")

    ratio = final / value
    return lambda i, total: value * ratio ** (i / max(total - 1, 1))
