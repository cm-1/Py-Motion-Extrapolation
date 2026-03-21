import re

import sympy as sp
from sympy import simplify
import matplotlib.pyplot as plt
import numpy as np

# For displaying math nicely
from IPython.display import display
sp.init_printing()

# F = 1/2 p v^2 C A
# F: drag force,
# p: fluid density (about 1.204 kg/m^3 for air at room temperature)
# v: speed
# A: cross-sectional area
# C: drag coefficient (dimensionless, generally lies in (~0.5, ~1.0)).
# Let's say that the mass of our objects tends to be (0.5, 1.5) kg, and that
# their cross-sectional area is (finish this later...)

# Now, F = ma. If we group all of our "constants" into the single constant c,
# then we get a = c v^2, c > 0. But this is, actually, just the acceleration 
# from drag; if we assume an initial acceleration, then a(t) = a_0 + c v(t)^2.
# Note: c > 0 from above, t >= 0 by definition, v0 > 0 because speed cannot be
# negative, and a speed of 0 means no drag.
c, t, v0 = sp.symbols("c, t, v_0", real=True, positive=True)
t = sp.symbols("t", real=True, negative=False)

a0 = sp.symbols("a_0", real=True, negative=True)
x0 = sp.symbols("x_0", real=True)
x = sp.Function('x')(t)
v = sp.Function('v')(t)

def solveDE(v_t_formula):
    global v
    global t
    global v0
    global x0
    de = sp.Eq(v.diff(t), v_t_formula)
    v_soln = sp.dsolve(de, v, ics={v.subs(t, 0): v0})
    vr = simplify(v_soln.rhs)
    a0_sub = sp.I * sp.sqrt(a0)

    # Because a0 is symbolic, the "== True" is NOT redundant! (a0 < 0) is not
    # a boolean if a0 is not created with "positive = True" or similar!
    if (a0 < 0) == True: 
        a0_sub = -a0_sub
    elif (a0 >= 0) != True:
        raise Exception("No information about a_0 sign known!") 
    vr2 = simplify(vr.subs({sp.sqrt(-a0): a0_sub}, simultaneous=True))
    x_int = simplify(sp.integrate(vr2, t))
    x_at_0 = x_int.subs({t: 0})
    x_final = simplify(x_int - x_at_0 + x0)
    return (vr2, x_final)


# Define the differential equation
hs_drag_a = -c * v**2
hs_vs1, hs_xs1 = solveDE(hs_drag_a)
hs_a_drag_a = a0 + hs_drag_a
hs_vs2, hs_xs2 = solveDE(hs_a_drag_a)

ls_drag_a = -c * v
ls_vs1, ls_xs1 = solveDE(ls_drag_a)
ls_a_drag_a = a0 + ls_drag_a
ls_vs2, ls_xs2 = solveDE(ls_a_drag_a)
j = sp.symbols("j", real = True)
ls_j_drag_a = ls_a_drag_a + j * t
ls_vs3, ls_xs3 = solveDE(ls_j_drag_a)


def deplot(f, t_f, a0_val, v0_val, c_val):
    subbed = simplify(f.subs({a0: a0_val, v0: v0_val, c: c_val}))
    print("Plotting:", subbed)
    fnp = sp.lambdify(t, subbed, "numpy")
    ts = np.linspace(0.001, t_f, 100)
    plt.plot(ts, fnp(ts))
    plt.show()

def geogebraStrFromStr(f_str):
    '''NOTE: Cannot handle nested parentheses well! Would need non-re fix!'''
    # Geogebra's order of operations will look at something like "a/c**2" and
    # graph (a/c)**2, whereas sympy means a/(c**2). So we need to be explicit.
    # ---
    # Below, we first look for division, since so far, that's the only thing
    # that's caused an order of operations problem with the exponents. Then
    # we check for either a variable or something in "(...)" being raised to
    # a different variable or something in "(...)".
    pattern = re.compile(
        r"(<?/\s*)(\b[\w\d]+|\([^)(]+\))\s*\*\*\s*([\w\d]+\b|\([^)(]+\))"
    )
    for find in pattern.finditer(f_str):
        f_str = f_str.replace(find.group(), "/(" + find.group()[1:] + ")")
    return f_str


def geogebraStrFromSymb(f, zero_x0 = True):
    global t
    global x0
    '''Get a string for Geogebra plots (or similar, like Wolfram):'''
    x_symb = sp.symbols("x", real=True)
    sub_dict = {t: x_symb}
    if zero_x0:
        sub_dict[x0] = 0
    f_str = str(f.subs(sub_dict))
    return geogebraStrFromStr(f_str)

