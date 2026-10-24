from abc import ABC, abstractmethod

from tp4.types.pattern import Pattern, Patterns

# Generic associative memory: stores patterns and answers a query with the stored one most alike.
# R is what the implementation reports for a query.
class AssociativeMemory[R](ABC):

    @abstractmethod
    def store(self, patterns: Patterns) -> None:
        pass

    @abstractmethod
    def recall(self, query: Pattern) -> R:
        pass
