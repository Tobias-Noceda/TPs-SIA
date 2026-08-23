from tp1.interfaces.state_manager import StateManager

def dfs[S](sm: StateManager[S], state: S) -> list[S] | None:
    if sm.is_goal(state):
        return [state]

    for next in sm.next(state):
        result = dfs(sm, next)

        if result is not None:
            result.append(state)
            return result

    return None
