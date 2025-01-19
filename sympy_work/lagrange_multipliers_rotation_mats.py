#%% The below code was originally written to do some work using Lagrange
#   Multipliers to investigate an equation involving rotation matrices.
#   I eventually found a working non-Lagrange approach, so I've removed most of
#   the problem-specific code that ended up not being useful, but I've kept the
#   various helper functions here in case they prove useful in the future.
import sympy as sp
from sympy import simplify

def getVecSymbols(vec_char, vec_column):
    symb_str = ""
    col_str = str(vec_column)
    for i in range(3):
        symb_str += vec_char + "_{{" + col_str + "}\\,{" + str(i) + "}}"
        if i < 2:
            symb_str += " "
    return sp.symbols(symb_str, real=True)

def vec9ToMat3(vec9):
    list2D = [[0 for _ in range(3)] for _ in range(3)]
    for r in range(3):
        for c in range(3):
            list2D[r][c] = simplify(vec9[3*c + r])
    return sp.Matrix(list2D)

def isZeroMat(mat):
    return simplify(sum(list(mat))) == 0

q0_vars = getVecSymbols('q', 0)
q1_vars = getVecSymbols('q', 1)
q2_vars = getVecSymbols('q', 2)

q0 = sp.Matrix(list(q0_vars))
q1 = sp.Matrix(list(q1_vars))
q2 = sp.Matrix(list(q2_vars))

Q = q0.row_join(q1).row_join(q2)

q_vec9 = q0.col_join(q1).col_join(q2)
Q_remade = vec9ToMat3(q_vec9)

if not isZeroMat(Q - Q_remade):
    raise Exception("Error in vec9 <-> mat3 code!")

#%% 
lams = sp.symbols("\\lambda_{0:3}")
ells = sp.symbols("\\ell_{0:3}")
unit_eqs = []
ortho_eqs = []
for i in range(3):
    qi = Q.col(i)
    q_next = Q.col((i + 1) % 3)
    unit_eqs.append(lams[i] * (1 - qi.dot(qi)))
    ortho_eqs.append(ells[i] * qi.dot(q_next))

# The below code is useless/meaningless for now. It's just to remind/demonstrate
# that list() can transform a sympy vector of equations into a Python list of
# them, and that sum() can be used to quickly convert a list of equations into
# a single one without a loop and "+="".
eqs_list = list(q_vec9) + unit_eqs
unit_eq_sum = sum(unit_eqs)

print("Done!")
