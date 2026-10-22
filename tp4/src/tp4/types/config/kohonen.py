from typing import TypedDict

from tp4.types.initialization_function import InitializationFunction
from tp4.types.neighborhood_function import NeighborhoodFunction
from tp4.types.similarity_function import SimilarityFunction
from tp4.types.topology import Topology

class KohonenDefinition(TypedDict):
    """
    Definition of the self-organizing map.

    Attributes:
        grid_size: k of the k × k grid of neurons.
        topology: Shape of the grid (rectangular or hexagonal).
        similarity: Picks the winner neuron of each sample.
        neighborhood: Update strength of the neurons around the winner.
        initialization: Draws the initial weights from the samples.
    """
    grid_size: int
    topology: Topology
    similarity: SimilarityFunction
    neighborhood: NeighborhoodFunction
    initialization: InitializationFunction
