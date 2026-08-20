# Data structures
import heapq
from collections import deque

# Models
from TP1.models.node import Node
from TP1.models.metadata import Metadata, Heuristic, MetaMapa

# Typings
from typing import Sequence, Mapping

type MetaMap[N: Node] = Mapping[Metadata[N], Metadata[N]]

def build_path[N: Node](came_from: MetaMapa[N], current: Metadata[N]) -> Sequence[N]:
    total_path = deque([current.node])

    while current.node in came_from:
        current = came_from[current]
        total_path.appendleft(current.node)

    return list(total_path)

# https://en.wikipedia.org/wiki/A*_search_algorithm
def a_star[N: Node](start: N, goal: N, h: Heuristic[N]) -> Sequence[N] | None:
    nodes_meta: Mapping[N, Metadata[N]] = {}

    open_nodes = [Metadata(start)]
    heapq.heapify(open_nodes)

    came_from: MetaMapa[N] = {}

    open_nodes[0].g_score = 0
    open_nodes[0].f_score = h(start)

    while len(open_nodes) != 0:
        if open_nodes[0].node == goal:
            return build_path(came_from, open_nodes[0])

        current = heapq.heappop(open_nodes)

        for peer in current.peers():
            neighbour = nodes_meta.setdefault(peer, Metadata(peer))

            if current.score(neighbour, h):
                came_from[neighbour] = current

                if neighbour not in open_nodes:
                    heapq.heappush(open_nodes, neighbour)

    return None

def main() -> None:
    print("Hello from sia-pseudo!")