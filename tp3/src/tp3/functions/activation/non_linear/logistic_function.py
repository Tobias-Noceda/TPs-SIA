from tp3.types.float import Float

import numpy as np

def logistic_function(x: Float, beta: Float = np.float64(1.0), L: Float = np.float64(1.0)) -> Float:
    """
    Compute the logistic function of x: 1 / (1 + exp(-2*beta*x)).

    Parameters:
    x : Input value.
    beta : float, optional
        The beta parameter for the logistic function.
        Default is 1.0.
    L : float, optional
        The L parameter for the logistic function.
        Default is 1.0.

    Returns:
        Logistic function applied to x.
    """
    return L / (1 + np.exp(-2 * beta * x))

def logistic_function_derivative(x: Float, beta: Float = np.float64(1.0), L: Float = np.float64(1.0)) -> Float:
    """
    Compute the derivative of the logistic function of x, with: 2 * beta * logistic(x, beta, L) * (1 - logistic(x, beta, L) / L).

    Parameters:
    x : Input value.
    beta : float, optional
        The beta parameter for the logistic function.
        Default is 1.0.
    L : float, optional
        The L parameter for the logistic function.
        Default is 1.0.

    Returns:
        Derivative of the logistic function applied to x.
    """
    o = logistic_function(x, beta, L)
    return 2 * beta * o * (1 - o / L)
