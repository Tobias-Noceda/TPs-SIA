from typing import Tuple

from tp3.types.float import Float
from tp3.types.classification_option import ClassificationOption

import numpy as np
import numpy.typing as npt

def _classify(predicted: npt.NDArray[Float], expected: npt.NDArray[Float]) -> ClassificationOption:
    """
    Classifies the predicted value against the expected value.

    Args:
        predicted (Float): The predicted value.
        expected (Float): The expected value.

    Returns:
        ClassificationOption: The classification result.
    """
    if np.argmax(predicted) == np.argmax(expected):
        return ClassificationOption.TRUE_POSITIVE
    return ClassificationOption.FALSE_POSITIVE

def digits_classification(predicted: Tuple[npt.NDArray[Float], ...], expected: Tuple[npt.NDArray[Float], ...]) -> Tuple[ClassificationOption, ...]:
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
