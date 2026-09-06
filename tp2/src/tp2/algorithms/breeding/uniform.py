from typing import Tuple

from tp2.types.individual import Individual

import random

def uniform_crossover(parents: Tuple[Individual, Individual]) -> Tuple[Individual, Individual]:
    p1, p2 = parents
    c1 = []
    c2 = []

    for i in range(len(p1)):
        if random.random() < 0.5:
            c1.append(p1[i])
            c2.append(p2[i])
        else:
            c1.append(p2[i])
            c2.append(p1[i])

    return c1, c2