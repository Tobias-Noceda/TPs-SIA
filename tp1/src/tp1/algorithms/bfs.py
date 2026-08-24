from typing import Sequence, Set, Deque
from collections import deque

from tp1.interfaces.state_manager import StateManager

# (current_state, previous_state)
type MetaTuple[S] = tuple[S, list[S]]

# @link https://en.wikipedia.org/wiki/Breadth-first_search
# @generic S The state type MUST be comparable and hashable
def bfs[S](sm: StateManager[S], start: S) -> tuple[Sequence[S] | None, int, int]:
    queue: Deque[MetaTuple[S]] = deque([(start, list())])
    visited: Set[S] = {start}
    expanded_nodes: Set[S] = set()

    while len(queue):
        state, path = queue.popleft()
        path.append(state)

        if sm.is_goal(state):
            return path, len(expanded_nodes), len(queue)

        for next in sm.next(state):
            if next not in visited:
                visited.add(next)
                queue.append((next, path.copy()))

        expanded_nodes.add(state)

    return None, len(expanded_nodes), 0
