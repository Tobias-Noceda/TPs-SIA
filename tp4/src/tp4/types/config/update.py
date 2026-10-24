import enum

from tp4.types.update_function import UpdateFunction

from tp4.functions.update.asynchronous import asynchronous_update
from tp4.functions.update.synchronous import synchronous_update

class UpdateRules(enum.Enum):
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"

    def function(self) -> UpdateFunction:
        functions: dict[UpdateRules, UpdateFunction] = {
            UpdateRules.SYNCHRONOUS: synchronous_update,
            UpdateRules.ASYNCHRONOUS: asynchronous_update,
        }

        return functions[self]
