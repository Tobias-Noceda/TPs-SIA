from typing import Tuple

from tp2.types.individual import Individual

import random

def single_point_crossover(parents: Tuple[Individual, Individual]) -> Tuple[Individual, Individual]:
    p1, p2 = parents

    point = random.randint(1, len(p1) - 1)

    c1 = tuple(p1[:point]) + tuple(p2[point:])
    c2 = tuple(p2[:point]) + tuple(p1[point:])

    return c1, c2
