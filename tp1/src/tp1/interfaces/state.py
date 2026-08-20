from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Sequence

@dataclass
class State(ABC):
    @abstractmethod
    def peers(self) -> Sequence[State]: pass

    @abstractmethod
    def dist(self, other: State) -> int: pass

    @abstractmethod
    def is_goal(self) -> bool: pass

    @abstractmethod
    def __eq__(self, other: object) -> bool: pass

    @abstractmethod
    def __hash__(self) -> int: pass

