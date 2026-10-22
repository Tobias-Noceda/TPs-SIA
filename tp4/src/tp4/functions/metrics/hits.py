import numpy as np
import numpy.typing as npt

# Samples mapped to each neuron
def hits(winners: npt.NDArray[np.intp], neurons: int) -> npt.NDArray[np.intp]:
    """Count of samples won by each neuron: (neurons,). Neurons with 0 hits are dead units."""
    return np.bincount(winners, minlength=neurons)
