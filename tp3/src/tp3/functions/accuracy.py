from tp3.types.test_result import TestResult
from tp3.types.float import Float

import numpy as np

def accuracy(result: TestResult) -> Float:
    """
    Computes the accuracy of the classification results.

    Args:
        result (TestResult): The test result containing classification outcomes.

    Returns:
        Float: The accuracy as a NumPy float value.
    """
    total = (
        result["true_positive"]
        + result["true_negative"]
        + result["false_positive"]
        + result["false_negative"]
    )
    correct = result['true_positive'] + result['true_negative']
    
    return np.float64(correct / total)

def precision(result: TestResult) -> float:
    """
    Computes the precision of the classification results.

    Args:
        result (TestResult): The test result containing classification outcomes.

    Returns:
        float: The precision as a float value.
    """
    true_positive = result["true_positive"]
    false_positive = result["false_positive"]
    
    return true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0

def recall(result: TestResult) -> float:
    """
    Computes the recall of the classification results.

    Args:
        result (TestResult): The test result containing classification outcomes.

    Returns:
        float: The recall as a float value.
    """
    true_positive = result["true_positive"]
    false_negative = result["false_negative"]
    
    return true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0

def f1_score(result: TestResult) -> float:
    """
    Computes the F1 score of the classification results.

    Args:
        result (TestResult): The test result containing classification outcomes.

    Returns:
        float: The F1 score as a float value.
    """
    p = precision(result)
    r = recall(result)
    
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0