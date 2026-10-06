from collections import Counter
from typing import Tuple

from tp3.types.classification_option import ClassificationOption
from tp3.types.test_result import TestResult

def confusion_matrix(classifications: Tuple[ClassificationOption, ...]) -> TestResult:
    """
    Counts the outcomes of a classification function into a TestResult, so the metrics in
    tp3.functions.accuracy can be computed from predictions without a perceptron.
    """
    counts = Counter(classifications)
    return {
        "true_positive": counts[ClassificationOption.TRUE_POSITIVE],
        "true_negative": counts[ClassificationOption.TRUE_NEGATIVE],
        "false_positive": counts[ClassificationOption.FALSE_POSITIVE],
        "false_negative": counts[ClassificationOption.FALSE_NEGATIVE],
    }
