import typing
from enum import Enum

class MOTION_MODEL(Enum):
    STATIC = 1
    VEL_DEG1 = 2
    VEL_DEG2 = 3
    ACC_DEG2 = 4
    JERK = 5
    CIRC_VEL_DEG1 = 6
    CIRC_VEL_DEG2 = 7
    CIRC_ACC = 8
    # MIN_JERK = 9
    # MIN_JERK_SPLIT = 10


class MOTION_DATA(Enum):
    LAST_BEST_LABEL_ONEHOT = 1
    TIMESTEP = 2

    VEL_DEG1_VEC3 = 3
    VEL_DEG2_VEC3 = 4
    ACC_VEC3 = 5
    JERK_VEC3 = 6
    JERK_ERR_VEC3 = 7
    CRACKLE_VEC3 = 8
    ROTATION_VEC3 = 9
    ROT_ACC_VEC3 = 10
    CIRC_VEL_DEG1_ERR_VEC3 = 11
    CIRC_VEL_DEG2_ERR_VEC3 = 12
    CIRC_ACC_ERR_VEC3 = 13

    CIRC_RAD = 14
    CIRC_SPEED = 15
    CIRC_ACC = 16
    CIRC_ANG_SPEED = 17
    CIRC_ANG_ACC = 18

    AX3_SQ_DIFF = 19

    DISP_MAG_DIFF = 20
    DISP_MAG_DIFF_TIMESCALED = 21
    DISP_MAG_RATIO = 22
    # BOUNCE_ANGLE = 21 # Redundant now with VEL_DEG1_VEC3 stuff.

    UNIT_ROT_AX_DIFF = 23
    UNIT_ROT_AX_DIFF_TIMESCALED = 24

    RAD_DIFF = 25
    TIMESCALED_RAD_DIFF = 26

    # Norms of vec6s composed of circle centres and radii-scaled normals.
    CIRC_VEC6_DIFF = 27

    TIME_SINCE_STATIONARY = 28
    TIME_SINCE_DIR_CHANGE = 29
    DIST_SINCE_DIR_CHANGE = 30

    TIME_CIRC_MOTION = 31
    ANG_SUM_CIRC_MOTION = 32
    DIST_SUM_CIRC_MOTION = 33

    CIRC_CENTRE_DIFF = 34

    FRAME_NUM = 35
    TIMESTAMP = 36

    PREV_FA_ANG_ACC = 37
    NEXT_FA_ANG_ACC = 38

    SPEED_ACC_RATIO = 39
    VEL_BCS_RATIOS = 40

    CURVATURE = 41
    LAST_CURVATURE = 42


    SPEED_JERK_RATIO = 43
    ACC_JERK_RATIO = 44
    SPEED_ORTHO_ACC_RATIO = 45
    CIRC_ACC_CIRC_SPEED_RATIO = 46
    CIRC_ANG_ACC_CIRC_ANG_SPEED_RATIO = 47
    CIRC_ANG_RATIO = 48

    BOUNCE_ANGLE_2_SUM = 49

    PLANE_NORMAL_DOT = 50

    DIST_FROM_CIRCLE = 51
    RATIO_FROM_CIRCLE = 52

    VEL_DEG2_MAG_DIFF = 53
    VEL_DEG2_MAG_DIFF_TIMESCALED = 54

    INV_DISP_MAG_RATIO = 55
    INV_VEL_BCS_RATIOS = 56
    INV_CIRC_ANG_RATIO = 57

    GT0 = 58
    GT1 = 59
    GT2 = 60
    GT3 = 61
    GT4 = 62
    GT5 = 63

    ROT_JERK_VEC3 = 64


    # VEL_DOT = 64                             # Units match work per kg (J/kg)


    # CURVATURE_V = 65
    # CURVATURE_A = 66
    # CURVATURE_J = 67
    # LAST_CURVATURE_V = 68
    # LAST_CURVATURE_A = 69
    # LAST_CURVATURE_J = 70


class OTHER_DIRECTION(Enum):
    ACC_ORTHO_DEG1 = 1
    PLANE_ORTHO = 2

RELATIVE_VECTOR = typing.Union[MOTION_DATA, OTHER_DIRECTION]
ALL_RELATIVE_VECTORS = (
    MOTION_DATA.VEL_DEG1_VEC3, MOTION_DATA.VEL_DEG2_VEC3, MOTION_DATA.ACC_VEC3,
    MOTION_DATA.JERK_VEC3, MOTION_DATA.JERK_ERR_VEC3, MOTION_DATA.ROTATION_VEC3,
    MOTION_DATA.ROT_ACC_VEC3, MOTION_DATA.ROT_JERK_VEC3,
    OTHER_DIRECTION.ACC_ORTHO_DEG1, OTHER_DIRECTION.PLANE_ORTHO
)

class OrthoVecDirPair(typing.NamedTuple):
    vec3: MOTION_DATA
    axis: OTHER_DIRECTION

ORTHO_VEC3_AX_PAIRS = [
    OrthoVecDirPair(MOTION_DATA.VEL_DEG1_VEC3, OTHER_DIRECTION.ACC_ORTHO_DEG1),
    OrthoVecDirPair(MOTION_DATA.VEL_DEG1_VEC3, OTHER_DIRECTION.PLANE_ORTHO),
    OrthoVecDirPair(MOTION_DATA.VEL_DEG2_VEC3, OTHER_DIRECTION.PLANE_ORTHO),
    OrthoVecDirPair(MOTION_DATA.ACC_VEC3, OTHER_DIRECTION.PLANE_ORTHO)
]

class ANG_OR_MAG(Enum):
    ANG = 1
    MAG_PROJ = 2
    MAG_DOT = 3

class SpecifiedMotionData(typing.NamedTuple):
    base_cat: MOTION_DATA
    axis: RELATIVE_VECTOR
    ang_or_mag: ANG_OR_MAG
    bidirectional: bool
    is_timestep_shifted: bool

    @property
    def name(self):
        ret_name = ""
        bn = self.base_cat.name 
        rn = self.axis.name
        last_underscore_ind = bn.rfind("_")
        if bn[last_underscore_ind:] != "_VEC3":
            raise ValueError("No \"VEC3\" found in base type {}!".format(bn))
        
        ret_name = bn[:last_underscore_ind] 

        dirs_eq = False
        if isinstance(self.axis, MOTION_DATA):
            if self.base_cat != self.axis:
                r_last_underscore_ind = rn.rfind("_")
                if rn[r_last_underscore_ind:] != "_VEC3":
                    raise ValueError(
                        "No \"VEC3\" found in relative key {}!".format(rn)
                    )
                ret_name += "_" + rn[:r_last_underscore_ind]
            else:
                dirs_eq= True
        else:
            ret_name += "_" + rn
        am = self.ang_or_mag.name
        if dirs_eq and not self.is_timestep_shifted:
            if self.ang_or_mag == ANG_OR_MAG.MAG_PROJ:
                am = "MAG"
            else:
                raise Exception("Redundant self-axis value!")
        ret_name += "_" + am

        if self.bidirectional:
            ret_name += "_BIDIR"
        if self.is_timestep_shifted:
            ret_name += "_SHIFT"
        return ret_name

class OneHotMotionData(typing.NamedTuple):
    base_cat: MOTION_DATA
    cat_num: int

    @property
    def name(self):
        bn = self.base_cat.name 
        last_underscore_ind = bn.rfind("_")
        if bn[last_underscore_ind:] != "_ONEHOT":
            raise ValueError("No \"ONEHOT\" found in base type {}!".format(bn))
        
        return bn[:(last_underscore_ind + 1)] + "CAT" + str(self.cat_num) 
    
MOTION_DATA_KEY_TYPE = typing.Union[
    MOTION_DATA, SpecifiedMotionData, OneHotMotionData
]
