# Data structures
from heapq import heappush, heappop
from collections import deque

# Typings
from typing import Sequence
from tp1.types.heuristic import Heuristic
from tp1.interfaces.state_manager import StateManager

type MetaTuple[S] = tuple[float, int, S]

# @link https://es.wikipedia.org/wiki/Algoritmo_voraz
# @generic S The state type MUST be comparable and hashable
def greedy[S](sm: StateManager[S], start: S, h: Heuristic[S]) -> Sequence[S] | None:
    counter: int = 0

    open_nodes: list[MetaTuple[S]] = [(h(start), counter, start)]
    closed_nodes: set[S] = set()

    previous_state: dict[S, S] = {}

    while open_nodes:
        peak: MetaTuple[S] = heappop(open_nodes)
        current: S = peak[2]

        if sm.is_goal(current):
            return build_path(previous_state, current)

        closed_nodes.add(current)

        for next in sm.next(current):
            if next in closed_nodes:
                continue

            counter += 1
            update: MetaTuple[S] = (sm.dist(current, next) + h(next), counter, next)

            heappush(open_nodes, update)
            previous_state[next] = current

    return None

def build_path[S](came_from: dict[S, S], current: S) -> Sequence[S]:
    total_path = deque([current])

    while current in came_from:
        current = came_from[current]
        total_path.appendleft(current)

    return list(total_path)
