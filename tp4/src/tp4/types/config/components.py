import enum

from tp4.types.component_selection import ComponentSelection

from tp4.functions.components.kaiser import kaiser_components
from tp4.functions.components.variance import variance_components
from tp4.functions.components.fixed import fixed_components

class ComponentCriteria(enum.Enum):
    KAISER = "kaiser"
    VARIANCE = "variance"
    FIXED = "fixed"

    def function(self, value: float | None) -> ComponentSelection:
        if self is ComponentCriteria.KAISER:
            return kaiser_components
        if value is None:
            raise ValueError(f"The '{self.value}' criterion requires a 'value'.")
        if self is ComponentCriteria.VARIANCE:
            return variance_components(float(value))
        return fixed_components(int(value))
