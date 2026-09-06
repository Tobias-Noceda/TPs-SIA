import bisect

from typing import Sequence, Tuple

from tp2.models.individual_fitness import IndividualFitness

# @link https://en.wikipedia.org/wiki/Fitness_proportionate_selection
def spin(
    population: Sequence[IndividualFitness],
    weights: Sequence[float],
    rolls: Sequence[float],
) -> Tuple[IndividualFitness, ...]:
    """
    Shared machinery behind every roulette-like selection method.

    This is NOT a Selection itself: roulette, universal, ranking and boltzmann all
    reduce to "assign a non negative weight to each individual and throw a list of
    numbers in [0, 1) at the accumulated distribution". They only differ in how the
    weights and the rolls are built, so that is the only thing they implement.

    Individual i owns the interval (q[i-1], q[i]] of the normalized accumulated
    weights, and a roll r selects the first i such that r < q[i].

    Args:
        population: the individuals to choose from.
        weights: one weight per individual, same order. Negative weights are clamped
            to 0. If every weight is 0 the distribution degenerates to uniform.
        rolls: the numbers in [0, 1) to look up. One selected individual per roll,
            so len(rolls) is the amount of selected individuals.

    Returns:
        The selected individuals, in roll order. Repetitions are expected and wanted:
        a fit individual is supposed to win several slots.
    """
    accumulated, total = _accumulate(weights)

    if total <= 0:
        # Every individual is worthless (or the population is empty of fitness):
        # fall back to a uniform distribution instead of dividing by zero.
        size = len(population)
        accumulated = [i + 1 for i in range(size)]
        total = float(size)

    last = len(population) - 1

    return tuple(
        # bisect_right, not bisect_left: an individual with weight 0 shares its
        # accumulated value with its predecessor and must never be selected.
        population[min(bisect.bisect_right(accumulated, roll * total), last)]
        for roll in rolls
    )

def _accumulate(weights: Sequence[float]) -> tuple[list[float], float]:
    """
    Accumulated (unnormalized) weights and their total.
    Kept unnormalized so the caller's rolls are scaled by the total instead,
    which saves a division per individual.
    """
    total = 0.0
    accumulated = []

    for weight in weights:
        total += max(weight, 0.0)
        accumulated.append(total)

    return accumulated, total
