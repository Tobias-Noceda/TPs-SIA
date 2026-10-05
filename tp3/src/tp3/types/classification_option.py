import enum

class ClassificationOption(enum.Enum):
    """Enum for the classification options."""
    TRUE_POSITIVE = 0
    TRUE_NEGATIVE = 1
    FALSE_POSITIVE = 2
    FALSE_NEGATIVE = 3 