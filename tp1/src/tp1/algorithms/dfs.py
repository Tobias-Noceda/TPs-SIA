from tp1.interfaces.state_manager import StateManager

visited = set()

# https://en.wikipedia.org/wiki/Depth-first_search
def dfs[S](sm: StateManager[S], state: S) -> list[S] | None:
    visited.add(state)

    if sm.is_goal(state):
        return [state]

    for next in sm.next(state):
        if next in visited:
            continue

        result = dfs(sm, next)
        if result is not None:
            result.append(state)
            return result

    return None
