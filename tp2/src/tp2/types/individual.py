from typing import Sequence

from tp2.interfaces.polygon import Polygon

# The sequence of polygons and its fitness value (float) is used to represent an individual in the genetic algorithm.
type Individual = Sequence[Polygon]
