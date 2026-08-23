from typing import Sequence

from tp1.interfaces.state_manager import StateManager

def iddfs[S](sm: StateManager[S], start: S, limit = 1_000) -> Sequence[S] | None:
    for depth in range(limit):
        result = dls(sm, start, depth)
        if result is not None:
            return result[::-1]
    return None

def dls[S](sm: StateManager[S], state: S, depth: int) -> list[S] | None:
    if depth == 0:
        return [state] if sm.is_goal(state) else None

    for next in sm.next(state):
        result = dls(sm, next, depth - 1)

        if result is not None:
            result.append(state)
            return result

    return None
