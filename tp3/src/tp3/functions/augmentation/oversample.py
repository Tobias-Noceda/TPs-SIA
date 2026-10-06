from tp3.types.float import Float

import numpy as np
import numpy.typing as npt

# Random oversampling of the minority classes
# @link https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis#Random_oversampling
def oversample(expected: npt.NDArray[Float]) -> npt.NDArray[np.intp]:
    """
    Indices of a training set where every class (one-hot `expected`) appears as often as the most
    frequent one: each class keeps all its samples and repeats random ones to fill the gap.
    """
    labels = expected.argmax(1)
    counts = np.bincount(labels)
    target = counts.max()
    indices = []
    for label, count in enumerate(counts):
        if count == 0:
            continue
        members = np.flatnonzero(labels == label)
        indices.append(members)
        indices.append(np.random.choice(members, target - count))
    return np.concatenate(indices)
