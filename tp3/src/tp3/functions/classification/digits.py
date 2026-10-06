from typing import Tuple

from tp3.types.float import Float
from tp3.types.classification_option import ClassificationOption

import numpy as np
import numpy.typing as npt

def _classify(predicted: npt.NDArray[Float], expected: npt.NDArray[Float]) -> ClassificationOption:
    """
    Compares the predicted digit (output neuron with the highest activation) with the expected
    one (position of the 1 in the one-hot label).

    Multi-class has no single positive class: a hit counts as TRUE_POSITIVE and a miss as
    FALSE_POSITIVE, so tp3.functions.accuracy.accuracy gives the fraction of correct digits.
    Per-digit precision/recall need a one-vs-rest confusion matrix instead.

    Args:
        predicted (npt.NDArray[Float]): The output layer activations.
        expected (npt.NDArray[Float]): The one-hot label.

    Returns:
        ClassificationOption: The classification result.
    """
    if np.argmax(predicted) == np.argmax(expected):
        return ClassificationOption.TRUE_POSITIVE
    return ClassificationOption.FALSE_POSITIVE

def digits_classification(predicted: Tuple[npt.NDArray[Float], ...], expected: Tuple[npt.NDArray[Float], ...]) -> Tuple[ClassificationOption, ...]:
    """
    Classification function for the 0-9 digits: argmax of the output layer vs. the one-hot label.

    Args:
        predicted (npt.NDArray[Float]): One output vector per sample.
        expected (npt.NDArray[Float]): One one-hot label per sample.
    """
    classifications: Tuple[ClassificationOption, ...] = tuple(
        _classify(pred, exp) for pred, exp in zip(predicted, expected)
    )
    return classifications
