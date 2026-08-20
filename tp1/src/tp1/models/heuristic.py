# Data structures
from tp1.src.tp1.models.state import State

# Typings
from typing import Callable

type Heuristic[S] = Callable[[S], int]
