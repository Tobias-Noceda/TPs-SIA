import enum

from tp4.types.topology import Topology

from tp4.functions.topology.rectangular import rectangular_positions
from tp4.functions.topology.hexagonal import hexagonal_positions

import numpy as np

class Topologies(enum.Enum):
    RECTANGULAR = "rectangular"
    HEXAGONAL = "hexagonal"

    def topology(self) -> Topology:
        topologies: dict[Topologies, Topology] = {
            # Squares of side 1, rotated to have flat sides
            Topologies.RECTANGULAR: Topology(rectangular_positions, 4, np.sqrt(2) / 2, np.pi / 4),
            # Pointy-top hexagons of width 1
            Topologies.HEXAGONAL: Topology(hexagonal_positions, 6, 1 / np.sqrt(3), 0.0),
        }

        return topologies[self]
