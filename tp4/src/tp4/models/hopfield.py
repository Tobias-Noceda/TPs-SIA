from dataclasses import dataclass, field
from typing import override

from tp4.interfaces.associative_memory import AssociativeMemory

from tp4.functions.energy.hopfield import hopfield_energy

from tp4.types.float import Float
from tp4.types.hopfield_recall import Convergence, HopfieldRecall, HopfieldStep
from tp4.types.pattern import Pattern, Patterns
from tp4.types.update_function import UpdateFunction

import numpy as np
import numpy.typing as npt
from numpy.random import RandomState

# Discrete Hopfield network: Hebbian weights and sign dynamics until the state is stable
# @link https://en.wikipedia.org/wiki/Hopfield_network
@dataclass
class Hopfield(AssociativeMemory[HopfieldRecall]):
    """
    N neurons, all connected to each other (not to themselves), with states ±1.

    Patterns and queries must be ±1 vectors of the same length N. The stored patterns are fixed
    points only if they are close to orthogonal and p ≲ 0.15 · N.
    """
    # One iteration S(t) → S(t + 1) (synchronous or asynchronous)
    update: UpdateFunction

    # Upper bound of iterations per query
    max_iterations: int

    # Order of the asynchronous updates, seeded from the config for reproducibility
    random: RandomState = field(default_factory=RandomState, repr=False, hash=False)

    # (N, N), symmetric with zero diagonal, set by store
    weights: npt.NDArray[Float] = field(init=False, repr=False, hash=False)

    @override
    def store(self, patterns: Patterns) -> None:
        """w_ij = (1/N) Σ_μ ξ_i^μ ξ_j^μ for j ≠ i, w_ii = 0."""
        patterns = self.__validate(patterns, 2)

        self.weights = patterns.T @ patterns / patterns.shape[1]
        np.fill_diagonal(self.weights, 0)

    @override
    def recall(self, query: Pattern) -> HopfieldRecall:
        """Evolve S(0) = ζ until S doesn't change, two states alternate or max_iterations."""
        if not hasattr(self, "weights"):
            raise RuntimeError("The patterns must be stored first.")

        state = self.__validate(query, 1)
        if len(state) != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} neurons, got {len(state)}.")

        steps = [HopfieldStep(0, state, hopfield_energy(self.weights, state))]
        convergence = Convergence.MAX_ITERATIONS

        for t in range(1, self.max_iterations + 1):
            state = self.update(self.weights, state, self.random)

            if np.array_equal(state, steps[-1].state):
                convergence = Convergence.STABLE
                break

            steps.append(HopfieldStep(t, state, hopfield_energy(self.weights, state)))

            if len(steps) >= 3 and np.array_equal(state, steps[-3].state):
                convergence = Convergence.CYCLE
                break

        return HopfieldRecall(tuple(steps), convergence)

    def __validate(self, values: npt.ArrayLike, ndim: int) -> npt.NDArray[Float]:
        values = np.array(values, dtype=np.float64)

        if values.ndim != ndim:
            raise ValueError(f"Expected {"a (patterns, neurons) matrix" if ndim == 2 else "a (neurons,) vector"}.")
        if not np.all(np.abs(values) == 1):
            raise ValueError("Every entry must be 1 or -1.")

        return values
