from enum import IntEnum

class EnvTypes(IntEnum):
    HABITAT = 1
    ROBOTHOR = 2
    NORMAL = 3
    DUP = 4
    REMOVE = 5
    LONGTAIL = 6

class ClassTypes(IntEnum):
    REGULAR = 1
    SPATIAL = 2
    APPEARENCE = 3
    HIDDEN = 4
    LONGTAIL = 5
    GPT = 6

class BackBones(IntEnum):
    B32 = 1
    B16 = 2
    L14 = 3

POSIBLE_CONFIGS = {
    EnvTypes.HABITAT : [ClassTypes.REGULAR, ClassTypes.GPT,],
    EnvTypes.ROBOTHOR : [ClassTypes.REGULAR, ClassTypes.GPT,],
    EnvTypes.NORMAL : [ClassTypes.SPATIAL, ClassTypes.APPEARENCE, ClassTypes.HIDDEN,],
    EnvTypes.DUP : [ClassTypes.SPATIAL, ClassTypes.APPEARENCE, ClassTypes.REGULAR],
    EnvTypes.REMOVE : [ClassTypes.HIDDEN,],
    EnvTypes.LONGTAIL: [ClassTypes.LONGTAIL,],
}