from typing import Sequence, Tuple
from interfaces.coords import Coords

# (PlayerCoords, [BoxCoords])
type BoardState = Tuple[Coords, Tuple[Coords]]
