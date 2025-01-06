#%% imports
from dataclasses import dataclass
import typing

import numpy as np

import posemath as pm


#%% Consts/defs for Black Hole Binary System test described in Boyle's paper.
'''
Boyle, M. The Integration of Angular Velocity. Adv. Appl. Clifford Algebras 27,
2345–2374 (2017). https://doi.org/10.1007/s00006-017-0793-z
'''

# To make sure we're not using too much memory during the simulation.
def FindGigsRequired(num_subintervals, num_steps):
    # There will be 3 non-const terms and their derivatives, two inverse
    # derivatives and inverses, 7 "saved" partial products that will be reused
    # in the derivative calculations (and the last of which is the gt rots),
    # and 2 temporary quat lists during derivative calculation calls.
    # Then finally there will be one set of final rotation predictions.
    # Each quaternion is 4 items.
    multiplier_for_quats = (3 + 3 + 2 + 2 + 7 + 2 + 1) * 4
    
    # We'll also have three angles per sample.
    doubles_per_sample = multiplier_for_quats + 3
    
    total_samples = (num_subintervals * num_steps) + 1

    total_bytes = doubles_per_sample * 8 * total_samples # 8bytes per double

    # Then there's the <= 11 intermediate quaternions generate for each
    # timesetp during the Runge Kutta process.
    total_bytes += 11 * 4 * 8 * num_steps

    return total_bytes / 1024**3

@dataclass
class BHB_Term:
    r_ind: int
    is_const: bool = False
    is_inv: bool = False

nda5 = typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
@dataclass
class BHB_Data:
    rs: nda5
    invs: nda5
    derivs: nda5
    inv_derivs: nda5
    left_calcs: typing.List[np.ndarray]
    composite_ang_vels: np.ndarray = None

# bhb_rmul = (BHB_Term(0, True), BHB_Term(1), BHB_Term(3, False, True))
bhb_rmul = (
    BHB_Term(0, True), BHB_Term(1), BHB_Term(4, True), BHB_Term(1, False, True),
    BHB_Term(3), BHB_Term(2), BHB_Term(3, False, True), BHB_Term(1)
)

omega_orb = 2*np.pi / 1000
omega_prec = omega_orb / 10
bhb_a = np.pi / 8
bhb_ap = 2*bhb_a / 100000
bhb_v = bhb_a / 10
bhb_angvel_mags = (omega_orb, bhb_ap, omega_prec)


bhb_r0_ha = -3*bhb_a / 20
bhb_r0_const = np.array([np.cos(bhb_r0_ha), np.sin(bhb_r0_ha), 0, 0])
bhb_r4_ha = bhb_v / 2
bhb_r4_const = np.array([np.cos(bhb_r4_ha), np.sin(bhb_r4_ha), 0, 0])

def timestepsBHB(start_time: int, end_time: int, num_subintervals: int,
                 reset_time: int):
    new_start = start_time % reset_time
    num_big_steps = (end_time - start_time)
    new_end = new_start + num_big_steps
    total_samples = (num_subintervals * num_big_steps) + 1

    return np.linspace(new_start, new_end, total_samples)


def bhbDataCalc(start_time: int, end_time: int, num_subintervals: int):

    r1_ha = timestepsBHB(start_time, end_time, num_subintervals, 1000)
    r2_ha = timestepsBHB(start_time, end_time, num_subintervals, 800000)
    r3_ha = timestepsBHB(start_time, end_time, num_subintervals, 10000)
    r1_ha *= (omega_orb/2)
    r2_ha = (bhb_a/2) + (bhb_ap/2) * r2_ha
    r3_ha *= (omega_prec/2)
    zeros = np.zeros(len(r1_ha))
    bhb_r1 = np.stack((np.cos(r1_ha), zeros, zeros, np.sin(r1_ha)), axis = -1)
    bhb_r2 = np.stack((np.cos(r2_ha), np.sin(r2_ha), zeros, zeros), axis = -1)
    bhb_r3 = np.stack((np.cos(r3_ha), zeros, zeros, np.sin(r3_ha)), axis = -1)

    bhb_rs = (bhb_r0_const, bhb_r1, bhb_r2, bhb_r3, bhb_r4_const)

    bhb_nonconst_axinds = (2, 0, 2)

    # Only R1 and R3 ever get inverted. Thus, we'll have "None" take the place of
    # the other elements.
    bhb_invs = (
        None, pm.conjugateQuats(bhb_r1), None, pm.conjugateQuats(bhb_r3), None
    )
    # We'll also use None as placeholders for items yet to be calculated
    bhb_derivs = [0.0, None, None, None, 0.0]
    for i in range(3):
        bhb_halfangvel = np.zeros(4)
        bhb_halfangvel[bhb_nonconst_axinds[i] + 1] = bhb_angvel_mags[i]/2
        bhb_derivs[i + 1] = pm.multiplyQuatLists(bhb_halfangvel, bhb_rs[i + 1])

    bhb_inv_derivs = [None, None, None, None, None]
    for i in (1, 3):
        bhb_inv_derivs[i] = -pm.multiplyQuatLists(
            bhb_invs[i], pm.multiplyQuatLists(bhb_derivs[i], bhb_invs[i])
        )

    left_calcs = [bhb_rs[bhb_rmul[0].r_ind]]
    for bhb_t in bhb_rmul[1:]:
        next_term = bhb_rs[bhb_t.r_ind]
        if bhb_t.is_inv:
            next_term = bhb_invs[bhb_t.r_ind]
        left_calcs.append(pm.multiplyQuatLists(left_calcs[-1], next_term))

    '''
    d/dt q0 q1 = q'0 q1 + q0 q'1 =  1/2 (a0 q0 q1 + q0 a1 q1)

    a01 = 2 (q'0 q1 + q0 q'1) (q0 q1)^-1
        = (a0 q0 q1 + q0 a1 q1) q1^-1 q0^-1
        = a0 + q0 a1 q0^-1
    '''
    c_ang_vel = np.zeros((len(bhb_rs[1]), 3))

    for term in bhb_rmul[::-1]:
        c_rot = None
        if not term.is_inv:
            c_rot = bhb_rs[term.r_ind]
        else:
            c_rot = bhb_invs[term.r_ind]
        c_ang_vel = pm.rotateVecsByQuats(c_rot, c_ang_vel)
        if not term.is_const:
            inner_ind = term.r_ind - 1
            curr_ang_vel = bhb_angvel_mags[inner_ind]
            if term.is_inv:
                curr_ang_vel = -curr_ang_vel
            c_ang_vel[:, bhb_nonconst_axinds[inner_ind]] += curr_ang_vel

    return BHB_Data(bhb_rs, bhb_invs, bhb_derivs, bhb_inv_derivs, left_calcs, c_ang_vel)

# Recursively calculate the derivative of these terms using the product rule.
def bhbQuatProdDeriv(terms: typing.List[BHB_Term], data: BHB_Data):
    left_deriv = None
    if len(terms) == 2:
        if not terms[0].is_const:
            if terms[0].is_inv:
                left_deriv = data.inv_derivs[terms[0].r_ind]
            else:
                left_deriv = data.derivs[terms[0].r_ind]
    else:
        left_deriv = bhbQuatProdDeriv(terms[:-1], data)
    
    right_deriv = None
    if not terms[-1].is_const:
        if terms[-1].is_inv:
            right_deriv = data.inv_derivs[terms[-1].r_ind]
        else:
            right_deriv = data.derivs[terms[-1].r_ind]
    left_d_pt = None
    right_d_pt = None
    if not (left_deriv is None):
        right_r = None
        if terms[-1].is_inv:
            right_r = data.invs[terms[-1].r_ind]
        else:
            right_r = data.rs[terms[-1].r_ind]
        left_d_pt = pm.multiplyQuatLists(left_deriv, right_r)
    if not (right_deriv is None):
        right_d_pt = pm.multiplyQuatLists(
            data.left_calcs[len(terms) - 2], right_deriv
        )
    if (left_d_pt is None) and (right_d_pt is None):
        return np.asarray(0.0)
    elif left_d_pt is None:
        return right_d_pt
    elif right_d_pt is None:
        return left_d_pt
    return left_d_pt + right_d_pt


#%% Calculations for the Black Hole Binary System.

steps = 500
subintervals = 3300

gigs_req = FindGigsRequired(subintervals, steps)

if gigs_req > 2.0:
    raise Exception("This will probably take too much memory!")

data = bhbDataCalc(0, steps, subintervals)

#%%
# derivs = bhbQuatProdDeriv(bhb_rmul, data)

# angvels_from_derivs = 2 * pm.multiplyQuatLists(derivs, pm.conjugateQuats(
#     data.left_calcs[-1]
# ))[:, 1:]

#%%
angVels = data.composite_ang_vels # angvels_from_derivs #
reorder_for_rk = np.empty((steps, subintervals + 1, 3))

reorder_for_rk[:, :subintervals] = angVels[:-1].reshape(steps, subintervals, 3)
reorder_for_rk[:, -1] = angVels[subintervals::subintervals]

#%%
startVals = data.left_calcs[-1][:-1:subintervals]
endVals = data.left_calcs[-1][subintervals::subintervals]
reconst = pm.integrateAngularVelocityRK(reorder_for_rk, startVals, 4)




d = np.abs(reconst - endVals)
print(d.max())

