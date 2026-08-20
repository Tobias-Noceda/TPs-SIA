# Data structures
from tp1.models.node import Node

# Typings
from typing import Callable

type Heuristic[N: Node] = Callable[[N], int]
type Goal[N: Node] = Callable[[N], bool]
