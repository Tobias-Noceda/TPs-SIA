from abc import ABC, abstractmethod

from tp4.types.samples import Samples

# Linear projection of the samples onto a few learned directions (e.g. principal components).
# fit learns the directions; project maps samples (P, n) onto them (P, q).
class Projection(ABC):

    @abstractmethod
    def fit(self, inputs: Samples) -> None:
        pass

    @abstractmethod
    def project(self, inputs: Samples) -> Samples:
        pass
