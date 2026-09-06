import random

from math import exp
from typing import Sequence, Tuple, Dict

from tp2.models.individual_fitness import IndividualFitness

from tp2.algorithms.selection._wheel import spin

# Below this the exponential overflows and the temperature is meaningless anyway.
MIN_TEMPERATURE = 1e-9

# @link https://en.wikipedia.org/wiki/Simulated_annealing
def boltzmann_selection(population: Sequence[IndividualFitness], data: Dict) -> Tuple[IndividualFitness, ...]:
    config = data["boltzmann"]
    k: int = config["select"]

    temperature = _temperature(config, data["generation"])

    scores = [individual.score for individual in population]
    hottest = max(scores)

    exponentials = [exp((score - hottest) / temperature) for score in scores]
    average = sum(exponentials) / len(exponentials)

    weights = [value / average for value in exponentials]
    rolls = [random.random() for _ in range(k)]

    return spin(population, weights, rolls)

def _temperature(config: Dict, generation: int) -> float:
    t0: float = config["t0"]
    tc: float = config["tc"]
    rate: float = config["k"]

    return max(tc + (t0 - tc) * exp(-rate * generation), MIN_TEMPERATURE)
