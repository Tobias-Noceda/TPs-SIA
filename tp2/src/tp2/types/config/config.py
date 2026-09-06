from typing import TypedDict

from tp2.types.config.breeding import BreedingAlgorithms
from tp2.types.config.fitness import FitnessAlgorithms
from tp2.types.config.mutation import MutationConfig
from tp2.types.config.selection import SelectionAlgorithms
from tp2.types.config.survival import SurvivalAlgorithms

class Config(TypedDict):
    breeding: BreedingAlgorithms
    fitness: FitnessAlgorithms
    mutation: MutationConfig
    selection: SelectionAlgorithms
    survival: SurvivalAlgorithms
    hyperparameters: dict
