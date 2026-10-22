import enum

from typing import Callable, TypedDict

from tp4.types.float import Float
from tp4.types.schedule_function import ScheduleFunction

from tp4.functions.schedule.constant import constant_schedule
from tp4.functions.schedule.inverse import inverse_schedule
from tp4.functions.schedule.linear import linear_schedule
from tp4.functions.schedule.exponential import exponential_schedule

import numpy as np

# A dict containing the hyperparameters of a schedule.
class ScheduleHyperparameters(TypedDict, total=False):
    final: float

class ScheduleFunctions(enum.Enum):
    CONSTANT = "constant"
    INVERSE = "inverse"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"

    def factory(self, value: float, hyperparameters: ScheduleHyperparameters | None) -> ScheduleFunction:
        v = np.float64(value)
        final = np.float64(hyperparameters.get("final", 0.0)) if hyperparameters and hyperparameters.get("final", None) is not None else None

        factories: dict[ScheduleFunctions, Callable[[Float, Float | None], ScheduleFunction]] = {
            ScheduleFunctions.CONSTANT: constant_schedule,
            ScheduleFunctions.INVERSE: inverse_schedule,
            ScheduleFunctions.LINEAR: linear_schedule,
            ScheduleFunctions.EXPONENTIAL: exponential_schedule,
        }

        return factories[self](v, final)
