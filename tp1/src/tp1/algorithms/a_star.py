# Definitions
from tp1.src.tp1.models.state import State

# Utils
import math

# Data structures
import heapq
from collections import deque

# Typings
from typing import Sequence
from tp1.models.heuristic import Heuristic, Goal

type MetaTuple[S: State] = tuple[float, int, S, float]

# https://en.wikipedia.org/wiki/A*_search_algorithm
def a_star[S: State](start: S, goal: Goal, h: Heuristic[S]) -> Sequence[S] | None:
    push = heapq.heappush
    pop = heapq.heappop

    counter: int = 0
    open_nodes: list[MetaTuple[S]] = [(h(start), counter, start, 0.0)]

    smallest_g_score: dict[S, float] = { start: 0.0 }

    came_from: dict[S, S] = {}

    while open_nodes:
        current: MetaTuple[S] = pop(open_nodes)

        node: S = current[2]
        gs: float = current[3]

        if smallest_g_score.get(node, math.inf) < gs:
            continue

        if goal(node):
            return build_path(came_from, node)

        peers: Sequence[S] = node.peers() # pyright: ignore[reportAssignmentType]
        for peer in peers:
            tentative: float = gs + node.dist(peer)

            if tentative < smallest_g_score.get(peer, math.inf):
                counter += 1
                update: MetaTuple[S] = (tentative + h(peer), counter, peer, tentative)

                push(open_nodes, update)
                smallest_g_score[peer] = tentative
                came_from[peer] = node

    return None

def build_path[N: State](came_from: dict[N, N], current: N) -> Sequence[N]:
    total_path = deque([current])

    while current in came_from:
        current = came_from[current]
        total_path.appendleft(current)

    return list(total_path)
