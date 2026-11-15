from typing import TypedDict

class VariationalDefinition(TypedDict):
    """
    Parameters of a variational autoencoder.

    Attributes:
        beta: λ of J = reconstruction + λ KL (1 = standard ELBO; > 1 pushes the codes towards N(0, I)).
    """
    beta: float
