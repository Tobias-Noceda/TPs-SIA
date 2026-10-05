import enum

from tp3.functions.classification.digits import digits_classification
from tp3.functions.classification.equals import equals_classification
from tp3.functions.classification.fraude import fraude_classification
from tp3.functions.classification.sign import sign_classification
from tp3.functions.classification.similar import similar_classification
from tp3.types.classification_function import ClassificationFunction

class ClassificationFunctions(enum.Enum):
    EQUALS = "equals"
    SIMILAR = "similar"
    SIGN = "sign"
    FRAUDE = "fraude"
    DIGITS = "digits"

    def function(self) -> ClassificationFunction:
        functions: dict[ClassificationFunctions, ClassificationFunction] = {
            ClassificationFunctions.EQUALS: equals_classification,
            ClassificationFunctions.SIMILAR: similar_classification,
            ClassificationFunctions.SIGN: sign_classification,
            ClassificationFunctions.FRAUDE: fraude_classification,
            ClassificationFunctions.DIGITS: digits_classification,
        }
        
        return functions[self]
