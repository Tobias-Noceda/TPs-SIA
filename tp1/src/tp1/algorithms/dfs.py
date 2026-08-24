from tp1.interfaces.state_manager import StateManager

visited = set()

# https://en.wikipedia.org/wiki/Depth-first_search
def dfs[S](sm: StateManager[S], state: S) -> list[S] | None:
    result = _dfs(sm, state)
    return result[::-1] if result is not None else None

def _dfs[S](sm: StateManager[S], state: S) -> list[S] | None:
    if sm.is_goal(state):
        return [state]

    visited.add(state)

    for next in sm.next(state):
        if next in visited:
            continue

        result = _dfs(sm, next)
        if result is not None:
            result.append(state)
            return result

    return None
