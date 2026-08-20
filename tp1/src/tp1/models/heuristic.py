# Data structures
from tp1.src.tp1.models.state import State

# Typings
from typing import Callable

type Heuristic[S: State] = Callable[[S], int]
type Goal[S: State] = Callable[[S], bool]
