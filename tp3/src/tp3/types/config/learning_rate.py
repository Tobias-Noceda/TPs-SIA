import enum

from tp3.types.learning_rate_function import LearningRateFactory
from tp3.types.config.lr_hyperparameters import LearningRateHyperparameters

from tp3.functions.learning_rate.constant import constant_learning_rate
from tp3.functions.learning_rate.momentum import momentum_learning_rate
from tp3.functions.learning_rate.adaptative import adaptative_learning_rate
from tp3.functions.learning_rate.adam import adam_learning_rate

class LearningRateFunctions(enum.Enum):
    CONSTANT = "constant"
    MOMENTUM = "momentum"
    ADAPTATIVE = "adaptative"
    ADAM = "adam"

    # Returns a factory, NOT a cached function (unlike activations): learning rate functions are
    # stateful, so each parameter array has to build its own
    def factory(self, learning_rate: float, hyperparameters: LearningRateHyperparameters | None) -> LearningRateFactory:
        hyper = hyperparameters or {}
        lr = float(learning_rate)

        match self:
            case LearningRateFunctions.CONSTANT:
                return lambda: constant_learning_rate(lr)
            case LearningRateFunctions.MOMENTUM:
                alpha = float(hyper.get("alpha", 0.9))
                return lambda: momentum_learning_rate(lr, alpha)
            case LearningRateFunctions.ADAPTATIVE:
                if "a" not in hyper or "b" not in hyper:
                    raise ValueError("The adaptative learning rate requires the hyperparameters 'a' and 'b'")
                a, b, k = float(hyper["a"]), float(hyper["b"]), int(hyper.get("k", 3))
                return lambda: adaptative_learning_rate(lr, a, b, k)
            case LearningRateFunctions.ADAM:
                beta1 = float(hyper.get("beta1", 0.9))
                beta2 = float(hyper.get("beta2", 0.999))
                epsilon = float(hyper.get("epsilon", 1e-8))
                return lambda: adam_learning_rate(lr, beta1, beta2, epsilon)
