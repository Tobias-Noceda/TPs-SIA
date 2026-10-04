from typing import TypedDict


# A dict containing the hyperparameters for the activation function.
class Hyperparameters(TypedDict, total=False):
    beta: float
    L: float
    