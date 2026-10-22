from tp4.types.float import Float
from tp4.types.schedule_function import ScheduleFunction

# Constant value over the whole training
def constant_schedule(value: Float, final: Float | None = None) -> ScheduleFunction:
    """value at every iteration (final is ignored)."""
    return lambda _, __: value
