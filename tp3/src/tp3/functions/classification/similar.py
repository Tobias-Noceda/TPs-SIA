from typing import Tuple

from tp3.types.float import Float
from tp3.types.classification_option import ClassificationOption

import numpy as np

def _classify(predicted: Float, expected: Float) -> ClassificationOption:
    """
    Classifies the predicted value against the expected value.
    Returns:
        ClassificationOption: The classification result.
    """
    if np.isclose(predicted, expected, atol=0.1):
        return ClassificationOption.TRUE_POSITIVE
    else:
        return ClassificationOption.FALSE_POSITIVE

def similar_classification(predicted: Tuple[Float, ...], expected: Tuple[Float, ...]) -> Tuple[ClassificationOption, ...]:
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
