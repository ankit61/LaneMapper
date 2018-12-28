import enum
import sys

class LIDARFactor(enum.Enum):
    DEPTH = enum.auto()
    REFLECTIVITY = enum.auto()

sys.modules[__name__] = LIDARFactor
