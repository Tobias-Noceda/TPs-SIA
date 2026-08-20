# Utils
import math

# Data structures
from heapq import heappush, heappop
from collections import deque

# Typings
from typing import Sequence
from interfaces.heuristic import Heuristic
from interfaces.state_manager import StateManager

type MetaTuple[S] = tuple[float, int, S, float]

# @link https://en.wikipedia.org/wiki/A*_search_algorithm
# @generic S The state type MUST be comparable and hashable
def a_star[S](sm: StateManager[S], start: S, h: Heuristic[S]) -> Sequence[S] | None:
    counter: int = 0
    open_nodes: list[MetaTuple[S]] = [(h(start), counter, start, 0.0)]

    smallest_g_score: dict[S, float] = { start: 0.0 }

    previous_state: dict[S, S] = {}

    while open_nodes:
        peak: MetaTuple[S] = heappop(open_nodes)

        curr: S = peak[2]
        gs: float = peak[3]

        if smallest_g_score.get(curr, math.inf) < gs:
            continue

        if sm.is_goal(curr):
            return build_path(previous_state, curr)

        for next in sm.next(curr):
            tentative: float = gs + sm.dist(curr, next)

            if tentative < smallest_g_score.get(next, math.inf):
                counter += 1
                update: MetaTuple[S] = (tentative + h(next), counter, next, tentative)

                heappush(open_nodes, update)
                smallest_g_score[next] = tentative
                previous_state[next] = curr

    return None

def build_path[S](came_from: dict[S, S], current: S) -> Sequence[S]:
    total_path = deque([current])

    while current in came_from:
        current = came_from[current]
        total_path.appendleft(current)

    return list(total_path)
