from typing import Tuple

from tp3.types.float import Float
from tp3.types.classification_option import ClassificationOption

def _classify(predicted: Float, expected: Float) -> ClassificationOption:
    """
    Classifies the predicted value against the expected value.

    Args:
        predicted (Float): The predicted value.
        expected (Float): The expected value.

    Returns:
        ClassificationOption: The classification result.
    """
    if predicted >= 0.85 and expected == 1.0:
        return ClassificationOption.TRUE_POSITIVE
    elif predicted < 0.85 and expected == 0.0:
        return ClassificationOption.TRUE_NEGATIVE
    elif predicted >= 0.85 and expected == 0.0:
        return ClassificationOption.FALSE_POSITIVE
    elif predicted < 0.85 and expected == 1.0:
        return ClassificationOption.FALSE_NEGATIVE
    else:
        raise ValueError(f"Invalid predicted ({predicted}) or expected ({expected}) values for classification.")

def fraude_classification(predicted: Tuple[Float, ...], expected: Tuple[Float, ...]) -> Tuple[ClassificationOption, ...]:
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
