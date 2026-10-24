from typing import List, TypedDict

from tp4.types.float import Float
from tp4.types.noise_function import NoiseFunction
from tp4.types.update_function import UpdateFunction

class HopfieldQuery(TypedDict):
    """
    Noisy queries built from a stored pattern.

    Attributes:
        pattern: Name of the stored pattern to corrupt.
        noise_type: Type of noise to apply to the query.
        noise: Fraction of flipped entries of each query.
    """
    pattern: str
    noise_function: NoiseFunction
    noise: List[Float]

class HopfieldDefinition(TypedDict):
    """
    Definition of the Hopfield network.

    Attributes:
        patterns: Paths of the pattern files to store.
        update: One iteration S(t) → S(t + 1) (synchronous or asynchronous).
        max_iterations: Upper bound of iterations per query.
        query: Noisy queries to recall.
    """
    patterns: List[str]
    update: UpdateFunction
    max_iterations: int
    query: HopfieldQuery
