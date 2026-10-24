import enum
from typing import NamedTuple, Tuple

from tp4.types.float import Float
from tp4.types.pattern import Pattern

class Convergence(enum.Enum):
    # S(t + 1) = S(t): fixed point (stored pattern, its inverse or a spurious state)
    STABLE = "stable"
    # S(t + 1) = S(t − 1) ≠ S(t): two states alternate forever (synchronous update only)
    CYCLE = "cycle"
    # The iteration limit was reached first
    MAX_ITERATIONS = "max_iterations"

class HopfieldStep(NamedTuple):
    """
    State of the network at a time step.

    Attributes:
        step: Time t (0 = the query).
        state: Copy of S(t) (N,).
        energy: H(S(t)).
    """
    step: int
    state: Pattern
    energy: Float

class HopfieldRecall(NamedTuple):
    """
    Evolution of the network for a query.

    Attributes:
        steps: S(0) = query, S(1), ..., S(t_final).
        convergence: Why the iteration stopped.
    """
    steps: Tuple[HopfieldStep, ...]
    convergence: Convergence

    @property
    def state(self) -> Pattern:
        """S(t_final): the pattern associated with the query."""
        return self.steps[-1].state
