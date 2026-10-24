from tp4.types.component_selection import ComponentSelection

import numpy as np

def variance_components(threshold: float) -> ComponentSelection:
    """Smallest q whose first q components explain at least `threshold` (in (0, 1]) of the total variance."""
    def select(eigenvalues):
        cumulative = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        # Tolerance so a threshold of 1 is reached despite rounding in the cumulative sum
        return int(np.argmax(cumulative >= threshold - 1e-12)) + 1

    return select
