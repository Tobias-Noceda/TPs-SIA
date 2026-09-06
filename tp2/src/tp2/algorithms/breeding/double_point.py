from typing import Tuple

from tp2.types.individual import Individual

import random

def double_point_crossover(parents: Tuple[Individual, Individual]) -> Tuple[Individual, Individual]:
    p1, p2 = parents

    point1, point2 = sorted(random.sample(range(1, len(p1)), 2))

    c1 = tuple(p1[:point1]) + tuple(p2[point1:point2]) + tuple(p1[point2:])
    c2 = tuple(p2[:point1]) + tuple(p1[point1:point2]) + tuple(p2[point2:])

    return c1, c2
