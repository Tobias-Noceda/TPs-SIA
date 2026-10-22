import enum

from tp4.types.similarity_function import SimilarityFunction

from tp4.functions.similarity.euclidean import euclidean_similarity
from tp4.functions.similarity.exponential import exponential_similarity

class SimilarityFunctions(enum.Enum):
    EUCLIDEAN = "euclidean"
    EXPONENTIAL = "exponential"

    def function(self) -> SimilarityFunction:
        functions: dict[SimilarityFunctions, SimilarityFunction] = {
            SimilarityFunctions.EUCLIDEAN: euclidean_similarity,
            SimilarityFunctions.EXPONENTIAL: exponential_similarity,
        }

        return functions[self]
