from typing import Tuple

from tp3.types.float import Float
from tp3.types.classification_option import ClassificationOption

# BigModel flags a transaction as fraud exactly when its probability is >= 0.85
DEFAULT_THRESHOLD = 0.85

def _classify(predicted: Float, expected: Float, threshold: float):
    """
    Classifies the predicted value against the expected value.

    Args:
        predicted (Float): The predicted value.
        expected (Float): The expected value.
        threshold (float): Predictions at or above it are considered fraud.

    Returns:
        ClassificationOption: The classification result.
    """
    if predicted >= threshold and expected == 1.0:
        return ClassificationOption.TRUE_POSITIVE
    elif predicted < threshold and expected == 0.0:
        return ClassificationOption.TRUE_NEGATIVE
    elif predicted >= threshold and expected == 0.0:
        return ClassificationOption.FALSE_POSITIVE
    elif predicted < threshold and expected == 1.0:
        return ClassificationOption.FALSE_NEGATIVE
    else:
        raise ValueError(f"Invalid predicted ({predicted}) or expected ({expected}) values for classification.")

def fraud_classification(predicted: Tuple[Float, ...], expected: Tuple[Float, ...], threshold: float = DEFAULT_THRESHOLD):
    """
    Classification function that determines if the predicted values indicate fraud.

    Args:
        predicted (npt.NDArray[Float]): An array of predicted values.
        expected (npt.NDArray[Float]): An array of expected values.
        threshold (float): Detection threshold, defaults to BigModel's own.
    """
    classifications: Tuple[ClassificationOption, ...] = tuple(
        _classify(pred, exp, threshold) for pred, exp in zip(predicted, expected)
    )
    return classifications
