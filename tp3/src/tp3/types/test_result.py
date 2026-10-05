from typing import TypedDict

class TestResult(TypedDict):
    """
    A dictionary representing the result of a training test.

    Attributes:
        true_positive (int): The number of true positives.
        true_negative (int): The number of true negatives.
        false_positive (int): The number of false positives.
        false_negative (int): The number of false negatives.

    """
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
