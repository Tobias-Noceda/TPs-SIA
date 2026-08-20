# Utils
import numpy as np

# Data structures
from functools import total_ordering
from tp1.models.node import Node

# Typings
from typing import Callable, Sequence

type Heuristic[N: Node] = Callable[[N], float]

@total_ordering
class Metadata[N: Node]:
    g_score: float = np.inf
    f_score: float = np.inf

    def __init__(self, node: N):
        self.node = node

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Metadata):
            return NotImplemented
        return self.node == other.node # type: ignore

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Metadata):
            return NotImplemented
        return self.f_score < other.f_score

    def peers(self) -> Sequence[N]:
        return self.node.peers() # type: ignore

    def score(self, other: Metadata[N], h: Heuristic[N]) -> bool:
        tentative = self.g_score + self.node.dist(other.node)
        cheaper = tentative < other.g_score

        if cheaper:
            other.g_score = tentative
            other.f_score = tentative + h(self.node)

        return cheaper