import enum

from tp2.interfaces.algorithms.fitness import ImageFitnessCalculator

from tp2.algorithms.fitness.delta import DeltaEFitnessCalculator
from tp2.algorithms.fitness.mixed import MixedFitnessCalculator
from tp2.algorithms.fitness.overlap import OverlapFitnessCalculator
from tp2.algorithms.fitness.ssim import SSIMFitnessCalculator

class FitnessAlgorithms(enum.Enum):
    DELTA_E = "delta"
    MIXED = "mixed"
    OVERLAP = "overlap"
    SSIM = "ssim"

    @property
    def calculator(self):
        calculators: dict[FitnessAlgorithms, type[ImageFitnessCalculator]] = {
            FitnessAlgorithms.DELTA_E: DeltaEFitnessCalculator,
            FitnessAlgorithms.MIXED: MixedFitnessCalculator,
            FitnessAlgorithms.OVERLAP: OverlapFitnessCalculator,
            FitnessAlgorithms.SSIM: SSIMFitnessCalculator,
        }
        
        return calculators[self]