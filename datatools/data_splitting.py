from enum import Enum

class DataSubsetKind(Enum):
    TRAIN = 1
    TEST = 2
    VALIDATION = 3
    WHOLE = 4

    @staticmethod
    def nonWholeValues():
        return (
            DataSubsetKind.TRAIN, DataSubsetKind.TEST, DataSubsetKind.VALIDATION
        )