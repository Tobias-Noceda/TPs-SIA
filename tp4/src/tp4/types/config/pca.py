from typing import TypedDict

from tp4.types.component_selection import ComponentSelection
from tp4.types.orientation_function import OrientationFunction

class PCADefinition(TypedDict):
    """
    Definition of the principal component analysis.

    Attributes:
        components: Picks how many components to keep from the sorted eigenvalues.
        orientation: Picks the sign of each eigenvector.
    """
    components: ComponentSelection
    orientation: OrientationFunction
