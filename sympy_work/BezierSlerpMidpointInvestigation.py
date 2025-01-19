import sympy as sp
from sympy import simplify

a = sp.symbols("a_{0:3}")#, real=True)
c = sp.symbols("c_{0:3}")#, real=True)
v = sp.symbols("v_{0:3}")#, real=True)
m = sp.symbols("m_{0:3}")#, real=True)

v_list = [v[0], v[1], v[2]]

a_ = sp.Matrix([a[0], a[1], a[2]])
c_ = sp.Matrix([c[0], c[1], c[2]])
v_ = sp.Matrix(v_list)
m_ = sp.Matrix([m[0], m[1], m[2]])

# Pythagorean quadruples
quadruples = [[1,0,0,1], [0,3,4,5], [1,2,2,3], [2,10,11,15], [4,13,16,21]] 
quadruple_vecs = []
for qu in quadruples:
    quadruple_vecs.append(sp.Matrix(qu).normalized())
    
def get_pyth_subs(vars_to_sub):
    ret_subs = dict()
    for i, vec in enumerate(vars_to_sub):
        for j in range(3):
            ret_subs[vec[j]] = quadruple_vecs[i][j]
    return ret_subs

def refl(vec, line_dir):
    return 2*vec.dot(line_dir)*line_dir - vec


b1 = refl(a_, v_)

v2 = refl(v_, m_)

b2 = simplify(refl(c_, v2))

my_b2_m = (8*m_.dot(c)*m_.dot(v_)**2 -4*m_.dot(v_)*v_.dot(c_))*m_ 
my_b2_v = (2*v_.dot(c_) - 4*m_.dot(v_)*m_.dot(c_))*v_
my_b2 = my_b2_m + my_b2_v - c_

'''
bdot = {mv[8(mc)(mv) -  4(vc)]m + [2vc - 4(mv)(mc)]v - c}{2(av)v - a}
bdot = (mv^2)[16(av)(mc)(mv) - 8(vc)(av)] + 4(av)(vc)vv - 8(mv)(mc)(av)vv ...

v2 = 
bdot = (2(av)v - a)(2(cv2)v2 - c) = 1

2(av)v - a = 2(cw)w - c
(av)v - (cw)w = (a - c)/2

(cw)w = (av)v + (c-a)/2
w = (mv)m - v


f(w1w1, w1w2, w1w3) = g(v1v1, v1v2, v2v3)


p(a,v) - p(c,w) = (a-c)/2

p(c,w) = p(a,v) + (c-a)/2

bdot = 4(av)(vv2)(cv2) - 2(av)vc - 2(cv2)(av2) + ac = 1

bdot = (2(av)v - a)(2(c[2(mv)m - v])[2(mv)m - v] - c)
'''

diff_test = sp.simplify(my_b2 - b2)
if diff_test != sp.Matrix([0,0,0]):
    raise Exception("My formulation was wrong!")
    
b_diff = b1 - b2

eqs = [b_diff[0], b_diff[1], b_diff[2], 1 - v_.dot(v_)]
simple_eqs = [simplify(e) for e in eqs]

#print("Starting sympy solve:")
#soln = sp.solve(simple_eqs, v_list, simplify=False, check=False)


pyth_subs = get_pyth_subs([a,c,m])   
pyth_eqs = [simplify(se.subs(pyth_subs)) for se in simple_eqs]

#soln = sp.solve(pyth_eqs)#, v_list, simplify=False, check=False)
nsoln_s = sp.nsolve(pyth_eqs, v_list, (0.2, 0.2, 0.2), verify=False)

#%%
# The below substitutions simplify with problem "without loss of generality",
# because we could just rotate the three points a, m, and c such that m is
# aligned with [1,0,0] and a lies on the unit sphere equator.
equator_subs = {m[0]:1, m[1]:0, m[2]:0, a[2]:0}
b_dot = b1.dot(b2)
b_dot_expand = sp.expand(b_dot.subs(equator_subs))
unit_len_subs = dict()
v0_sq_sub = 1 - v[2]**2 - v[1]**2
unit_len_subs[v[0]**2] = v0_sq_sub
unit_len_subs[v[0]**3] = v[0] * unit_len_subs[v[0]**2]
b_dot_simpler = simplify(b_dot_expand.subs(unit_len_subs))

# Now, unfortunately, we must turn to sqrts in order to turn this into a
# function of two variables that we can graph for insight...
b_dot_2var = simplify(b_dot_simpler.subs({v[0]: sp.sqrt(v0_sq_sub)}))

#%%
import matplotlib.pyplot as plt
import numpy as np
from posemath import quatBezier

def getRandomVecSubs():
    a_r = np.random.uniform(-1, 1, 3)
    a_r[2] = 0 
    a_r /= np.linalg.norm(a_r)
    c_r = np.random.uniform(-1, 1, 3)
    c_r /= np.linalg.norm(c_r)
    ret_val = dict()
    for i in range(3):
        ret_val[a[i]] = a_r[i]
        ret_val[c[i]] = c_r[i]
    return a_r, c_r, ret_val


def refl_np(vec, line_dir):
    return 2*np.dot(vec, line_dir)*line_dir - vec

def getLineCoords(pt0, pt1):
    return np.stack((pt0, pt1), axis=-1)

a_r, c_r, np_rand_subs = getRandomVecSubs()
b_dot_2var_rand = simplify(b_dot_2var.subs(np_rand_subs))
b_dot_np = sp.lambdify([v[1], v[2]], b_dot_2var_rand, modules=['numpy'])

grid_w = 100
grid = 1.0 - (2.0 / grid_w)*np.mgrid[:(grid_w + 1), :(grid_w + 1)]
grid = grid.reshape(2, -1)
b_np_zs = b_dot_np(grid[0], grid[1])

np_max_thresh = 0.99
b_np_zs_9 = b_np_zs > np_max_thresh
b_np_max_maybes = b_np_zs.copy()

b_np_maxes = []
while np.sum(b_np_max_maybes > np_max_thresh) > 0:
    max_loc = np.nanargmax(b_np_max_maybes)
    v1np = grid[0][max_loc]
    v2np = grid[1][max_loc]
    b_np_maxes.append([np.sqrt(1 - v2np*v2np - v1np*v1np), v1np, v2np])
    
    # Remove all max candidates that are super close to this one
    np_dists = grid.transpose() - np.array([v1np, v2np])
    np_dists = np.linalg.norm(np_dists, axis = -1)
    b_np_max_maybes[np_dists < 0.2] = 0
    

    
b_np_maxes = np.array(b_np_maxes)

b_np_zs_2D = b_np_zs.reshape(grid_w + 1, grid_w + 1)
grid_xs_2D = grid[0].reshape(grid_w + 1, grid_w + 1)
grid_ys_2D = grid[1].reshape(grid_w + 1, grid_w + 1)

fig = plt.figure(figsize=plt.figaspect(1/2))

# Plot points in normal "orthonormal" space.
ax = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(grid_xs_2D, grid_ys_2D, b_np_zs_2D)
# ax.scatter(b_np_maxes[:, 1], b_np_maxes[:, 2], 1, color='red')
''
bez_us = np.linspace(0, 1, 50)

colours = ['green', 'brown', 'black', 'pink', 'cyan', 'purple', 'lime']

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.scatter(*(a_r.reshape(3,1)), color = 'red')
ax2.scatter(*(c_r.reshape(3,1)), color = 'blue')
ax2.scatter([1], [0], [0], color = 'yellow')
ax2.scatter([-1], [0], [0], color = 'orange')
for i, v_opt in enumerate(b_np_maxes):
    col = colours[i % len(colours)]
    ax.scatter([v_opt[1]], [v_opt[2]], [1.0], color = col)
    ax2.plot(*getLineCoords(-v_opt, v_opt), color=col)
    v2_opt = refl_np(v_opt, np.array([1.0, 0.0, 0.0]))
    ax2.plot(*getLineCoords(-v2_opt, v2_opt), color=col, linestyle='dashed')
    b_opt = refl_np(a_r, v_opt)
    ax2.scatter(*(b_opt.reshape(3,1)), color = col)
    ctrl_pts = [a_r.reshape(1,3), b_opt.reshape(1,3), c_r.reshape(1,3)]
    bez_pts = np.array([quatBezier(ctrl_pts, u).flatten() for u in bez_us])
    ax2.plot(bez_pts[:, 0:1], bez_pts[:, 1:2], bez_pts[:, 2:], color = col)
plt.show()

#%%
# For a quadratic Bezier curve in Euclidean space, m = 0.5(b + 0.5(a+c))
# The below is to test if similar occurs; does the arc between (a+c)/|a+c|
# and m intersect b? Because that would probably make things MUCH easier....
# ... and now that I've run the below code, it seems that doesn't hold true.
# Oh well. So much for yet another idea...W

# Produce a situation with known b, which we use to derive m:
b_symb = sp.symbols("b_{0:3}", real=True)
b_ = sp.Matrix([b_symb[0], b_symb[1], b_symb[2]])

left_mid = (a_ + b_).normalized()
right_mid = (b_ + c_).normalized()
mid_mid = (left_mid + right_mid).normalized()

slerp_subs = get_pyth_subs([a, b_symb, c])

mid_mid_s = simplify(mid_mid.subs(slerp_subs))
    
a_c_sum = simplify((a_ + c_).subs(slerp_subs))
mac_cross = simplify(a_c_sum.cross(mid_mid_s))
b_s = b_.subs(slerp_subs)
b_planar_test = simplify(b_s.dot(mac_cross))

#%%



a_s = a_.subs(slerp_subs)
b_s = b_.subs(slerp_subs)
c_s = c_.subs(slerp_subs)

left_mid_s = left_mid.subs(slerp_subs)
right_mid_s = right_mid.subs(slerp_subs)

left_test = simplify(left_mid_s / sp.sqrt(2))
right_test = simplify(right_mid_s / sp.sqrt(870))

for i in range(3):
    slerp_subs[m[i]] = mid_mid_s[i]
    
slerp_eqs = [simplify(se.subs(slerp_subs)) for se in simple_eqs]

known_soln_subs = dict()
for i in range(3):
    known_soln_subs[v[i]] = left_mid_s[i]

slerp_eqs_subs = []
for sle in slerp_eqs:
    slerp_eqs_subs.append(simplify(sle.subs(known_soln_subs)))
    
nsoln_subs = get_pyth_subs([a, c, m])
for i in range(3):
    nsoln_subs[v[i]] = nsoln_s[i]

nsoln_test = [pe.subs(nsoln_subs).evalf() for pe in pyth_eqs]


'''
mL = (a+b)/|a+b|
mR = (b+c)/|b+c|
mm = normalize((a+b)|b+c| + (b+c)|a+b|)

mm =p (a+b)|b+c| + (b+c)|a+b|
   =  (a+b)sqrt(2 + 2b*c) + (b+c)sqrt(2 + 2a*b)
   =p (a+b)sqrt(1 + b*c) + (b+c)sqrt(1 + a*b)
   =  (aj+bj)sqrt(1 + S bici) + (bj + cj)sqrt(1 + S aibi)
   =  sqrt( (aj + bj)^2 (1 + S bici)) + sqrt((bj + cj)^2 (1 + S aibi))
   
mm =p (a+b) + (b+c)|a+b|/|b+c|
mm =p (a+b) + (b+c)sqrt(2)sqrt[(1 + a*b)/(1 + c*b)]

mm*(a+b)/|a+b| = mm*(c+b)/|c+b|


|a+b| = |b+c-(c-a)| = sqrt(|b+c|^2 + |c-a|^2 - 2(b+c)*(c-a))

Ra + Qc =p m
RRa = QQc

QRa + QQc =p Qm
QRa + RRa =p Qm
Ra + QtRRa =p m; let W = QtRR
(R + W)a =p m

m = Se(Se(a,b), Se(b,c))
Ise(m) = 
'''

#%%
soln_s = sp.solve(slerp_eqs, v_list, check=False, simplify=False)

#soln = sp.solve_poly_system(simple_eqs, v[0], v[1], v[2])

#   solve
#   nonlinsolve
#   check=False, simplify=False
#   Remove real=True from symbols
#   solveset (one var)
#   Wolfram alpha
#   Pythagorean triples?
#Solve linear version, then try solving nonlinear?

# https://docs.sympy.org/latest/modules/solvers/solvers.html#systems-of-polynomial-equations
