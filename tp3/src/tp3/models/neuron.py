from dataclasses import dataclass

from tp3.types.activation_function import ActivationFunction, DerivativeActivationFunction

@dataclass
class Neuron:
    activation_function: ActivationFunction
    derivative_activation_function: DerivativeActivationFunction

    def activate(self, x):
        return self.activation_function(x)

    def derivate(self, x):
        return self.derivative_activation_function(x)
