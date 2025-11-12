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

    @staticmethod
    def fromName(name: str):
        upper_name = name.upper()
        for _dsk in DataSubsetKind:
            if _dsk.name == upper_name:
                return _dsk
        raise ValueError(upper_name + " is not a valid DataSubsetKind!")
