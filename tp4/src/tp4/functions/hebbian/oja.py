from tp4.types.float import Float
from tp4.types.samples import Sample

import numpy.typing as npt

# Oja's rule: Hebbian update with a decay term that keeps ‖w‖ bounded (it tends to 1)
# @link https://en.wikipedia.org/wiki/Oja%27s_rule
def oja(weights: npt.NDArray[Float], sample: Sample, output: Float, learning_rate: Float) -> npt.NDArray[Float]:
    """Δw = η · s · (x − s · w), with s = x · w the output of the neuron: (n,)."""
    return learning_rate * output * (sample - output * weights)
