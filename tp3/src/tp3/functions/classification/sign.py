from typing import Tuple

from tp3.types.float import Float
from tp3.types.classification_option import ClassificationOption

import numpy as np

def _classify(predicted: Float, expected: Float):
    """
    Classifies the predicted value against the expected value.
    Returns:
        ClassificationOption: The classification result.
    """
    if np.sign(predicted) == np.sign(expected):
        return ClassificationOption.TRUE_POSITIVE
    else:
        return ClassificationOption.FALSE_POSITIVE

def sign_classification(predicted: Tuple[Float, ...], expected: Tuple[Float, ...]):
    """
    Classification function that determines if the predicted values indicate fraud.

    Args:
        predicted (npt.NDArray[Float]): An array of predicted values.
        expected (npt.NDArray[Float]): An array of expected values.
    """
    classifications: Tuple[ClassificationOption, ...] = tuple(
        _classify(pred, exp) for pred, exp in zip(predicted, expected)
    )
    return classifications
