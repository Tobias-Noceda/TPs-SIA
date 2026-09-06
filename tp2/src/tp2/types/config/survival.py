import enum

from tp2.interfaces.algorithms.survival import Survival

from tp2.algorithms.survival.additive import additive_survival
from tp2.algorithms.survival.exclusive import exclusive_survival

class SurvivalAlgorithms(enum.Enum):
    ADDITIVE = "additive"
    EXCLUSIVE = "exclusive"

    @property
    def function(self) -> Survival:
        functions: dict[SurvivalAlgorithms, Survival] = {
            SurvivalAlgorithms.ADDITIVE: additive_survival,
            SurvivalAlgorithms.EXCLUSIVE: exclusive_survival,
        }

        return functions[self]