from tp4.types.component_selection import ComponentSelection

def fixed_components(count: int) -> ComponentSelection:
    """Always `count` components (capped at the number of variables)."""
    return lambda eigenvalues: min(count, len(eigenvalues))
