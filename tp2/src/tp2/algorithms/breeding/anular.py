from typing import Tuple

from tp2.types.individual import Individual

import random

def anular_crossover(parents: Tuple[Individual, Individual]) -> Tuple[Individual, Individual]:
    p1, p2 = parents

    start = random.randint(0, len(p1) - 1)
    length = random.randint(1, len(p1) - 1)

    c1 = list(p1)
    c2 = list(p2)

    for i in range(length):
        index = (start + i) % len(p1)
        c1[index], c2[index] = c2[index], c1[index]

    return tuple(c1), tuple(c2)