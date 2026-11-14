from typing import TypedDict


# A dict containing the hyperparameters for the learning rate (optimizer).
class LearningRateHyperparameters(TypedDict, total=False):
    alpha: float    # momentum
    a: float        # adaptative: additive increase
    b: float        # adaptative: multiplicative decrease
    k: int          # adaptative: epochs in a row before adjusting
    beta1: float    # adam
    beta2: float    # adam
    epsilon: float  # adam
