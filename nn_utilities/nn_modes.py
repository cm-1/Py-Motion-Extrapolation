from enum import Enum

class OutVecMode(Enum):
    JAV_MULTIPLIERS = 1
    VEL_ALIGNED_VEC3 = 2
    WORLD_VEC3 = 3
    WORLD_DISP = 4
    ROT_ALIGNED_VEC3 = 5
    ROT_AA = 6
    ROT_VEL_AA = 7
    ROT_FIXED_AX = 8
    LAGRANGE_POLY = 9
    
