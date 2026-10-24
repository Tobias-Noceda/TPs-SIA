from tp4.types.float import Float

import numpy as np
import numpy.typing as npt

def cosine_similarity(a: npt.NDArray[Float], b: npt.NDArray[Float]) -> Float:
    """cos of the angle between two vectors: a · b / (‖a‖ ‖b‖), in [-1, 1]."""
    return np.float64(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
