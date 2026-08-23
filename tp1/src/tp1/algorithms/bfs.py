from typing import Sequence
from collections import deque

from tp1.interfaces.state_manager import StateManager

# (current_state, previous_state)
type MetaTuple[S] = tuple[S, list[S]]

# @link https://en.wikipedia.org/wiki/Breadth-first_search
# @generic S The state type MUST be comparable and hashable
def bfs[S](sm: StateManager[S], start: S) -> Sequence[S] | None:
    queue: deque[MetaTuple[S]] = deque()
    queue.append((start, list()))

    closed_nodes: set[S] = {start}

    while len(queue):
        state, path = queue.popleft()
        path.append(state)

        if sm.is_goal(state):
            return path

        for next in sm.next(state):
            if next not in closed_nodes:
                closed_nodes.add(next)
                queue.append((next, list(path)))

    return None
