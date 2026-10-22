from typing import Callable, NamedTuple

from tp4.types.grid import GridPositions

class Topology(NamedTuple):
    """
    Shape of the map grid.

    Attributes:
        positions: Places the neurons of a k × k grid, adjacent neurons at distance 1.
        vertices: Vertices of the cell drawn around each neuron (4 = square, 6 = hexagon).
        cell_radius: Circumradius of that cell so cells tile the plane.
        orientation: Rotation (radians) of that cell.
    """
    positions: Callable[[int], GridPositions]
    vertices: int
    cell_radius: float
    orientation: float
