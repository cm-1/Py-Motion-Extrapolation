import sympy as sp
from sympy import simplify, Derivative

# import matplotlib as plt

# For displaying math nicely
from IPython.display import display
sp.init_printing()

ys = sp.IndexedBase("y") # Sampled positions.
yps = sp.IndexedBase("y'") # y primes, for velocities I guess.
# Var not named "i" so I can still use "i" in for loops:
sp_i = sp.symbols('i', cls=sp.Idx) 
x = sp.symbols("x")

# Assumes uniform spacing of sample values' x (or time) coordinates.
def LagrangePoly(deg, data_vars = ys, shift = 0):
    res = 0
    for i in range(deg + 1):
        part = data_vars[i + shift]
        for j in range(deg + 1):
            if i != j:
                part *= (x - j)/(i - j)
        res += part
    return simplify(res)


lag2 = LagrangePoly(2)
lag2_d = Derivative(lag2, x)

yp_subs = dict()
for i in range(3):
    yp_subs[yps[i]] = simplify(lag2_d.subs({x:i}))
    display(sp.Eq(yps[i], yp_subs[yps[i]]))

lag2p = LagrangePoly(2, yps).subs(yp_subs)

lag3 = LagrangePoly(3)

#%%

def constAccelForDeg(deg):
    lag_deg = LagrangePoly(deg)
    lag_d = simplify(Derivative(lag_deg, x))
    lag_d_at_end = lag_d.subs({x:deg})
    display(lag_d_at_end)
    lag_d2 = simplify(Derivative(lag_d, x))
    lag_acc_at_end = lag_d2.subs({x:deg})
    display(lag_acc_at_end)

    print("Coord for const-accel deg=3 calculation:")
    display(simplify(ys[deg] + lag_d_at_end + (lag_acc_at_end/2)))
constAccelForDeg(4)
#%%
# The below was testing what happens when you model velocity using a quadratic
# fit of the last velocities but you initialize by doing a quadratic fit of the
# last positions. Of course, when initialized this way, the velocities will just
# lie on a line, but my plan was to investigate what happens when the 
# initialization result gets altered by a process like Kalman filtering.
# Didn't write the code for all that yet; TODO, I guess.
display(simplify(Derivative(lag3, x).subs({x:4})))

lag2p_integ = sp.integrate(lag2p, (x, 2, 3))
integ_res = ys[2] + lag2p_integ
display(integ_res)


yp_subs[yps[3]] = lag2p.subs({x:3})

display(yp_subs[yps[3]])

lag2p_s1 = LagrangePoly(2, yps, 1).subs(yp_subs)

lag2p_s1_integ = sp.integrate(lag2p_s1, (x, 2, 3))

newThing = simplify(integ_res + lag2p_s1_integ)

display(newThing)

