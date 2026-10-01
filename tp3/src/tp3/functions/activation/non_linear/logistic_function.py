from tp3.types.float import Float

import numpy as np

def logistic_function(x: Float) -> Float:
    """
    Compute the logistic function of x, with beta=1: 1 / (1 + exp(-2x)).

    Parameters:
    x : array-like
        Input value(s).

    Returns:
    array-like
        Logistic function applied to each element of x.
    """
    return 1 / (1 + np.exp(-2 * 1 * x)) # Beta = 1 explisitly

def logistic_function_derivative(x: Float) -> Float:
    """
    Compute the derivative of the logistic function of x, with beta=1: 2 * logistic(x) * (1 - logistic(x)).

    Parameters:
    x : array-like
        Input value(s).

    Returns:
    array-like
        Derivative of the logistic function applied to each element of x.
    """
    o = logistic_function(x)
    return 2 * o * (1 - o)
