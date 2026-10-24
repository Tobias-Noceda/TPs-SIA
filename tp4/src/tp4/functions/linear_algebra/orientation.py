from tp4.types.float import Float

import numpy as np
import numpy.typing as npt

def max_loading_positive(eigenvectors: npt.NDArray[Float]) -> npt.NDArray[Float]:
    """Flip each column so that its entry of largest absolute value is positive: deterministic signs across runs."""
    largest = eigenvectors[np.argmax(np.abs(eigenvectors), axis=0), np.arange(eigenvectors.shape[1])]
    return eigenvectors * np.where(largest < 0, -1.0, 1.0)

def keep_orientation(eigenvectors: npt.NDArray[Float]) -> npt.NDArray[Float]:
    """Keep the signs returned by the eigen solver."""
    return eigenvectors

def align(vector: npt.NDArray[Float], reference: npt.NDArray[Float]) -> npt.NDArray[Float]:
    """vector or −vector, whichever points to the same side as reference (non-negative dot product)."""
    return -vector if vector @ reference < 0 else vector
