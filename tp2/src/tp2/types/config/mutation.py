import enum

from typing import TypedDict

from tp2.interfaces.algorithms.mutation import Mutation

from tp2.algorithms.mutation.gene_mutation import gene_mutation
from tp2.algorithms.mutation.uniform_mutation import uniform_mutation
from tp2.algorithms.mutation.complete_mutation import complete_mutation

class MutationAlgorithms(enum.Enum):
    GENE = "gene"
    UNIFORM = "uniform"
    COMPLETE = "complete"

    @property
    def function(self) -> Mutation:
        functions: dict[MutationAlgorithms, Mutation] = {
            MutationAlgorithms.GENE: gene_mutation,
            MutationAlgorithms.UNIFORM: uniform_mutation,
            MutationAlgorithms.COMPLETE: complete_mutation,
        }

        return functions[self]

class MutationConfig(TypedDict):
    algorithm: MutationAlgorithms
    rate: float | None