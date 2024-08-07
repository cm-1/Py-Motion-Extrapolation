# This code was to figure out whether the Hou and Zhu paper should be
# interpreted in one particular way or another.

import sympy
from sympy import simplify
from IPython.display import display
import matplotlib
from sympy import pprint

sympy.init_printing()

uprint = lambda x: pprint(x, use_unicode=True)
uprints = lambda x: pprint(simplify(x), use_unicode=True)

x, v, A, B, U, C, K, z = sympy.symbols("x, v, A, B, U, C, K, z", commutative=False)
H = sympy.Symbol("H", commutative=False)

def xNext1(xc):
    y = z - H*xc
    xi = xc + K * y
    return A * xi + B*U
    
def xNext2(xc):
    y = z - H*xc
    xi = xc + K * y
    return A * xi
    
    
def rxNext2(n):
    res = xNext2(x)
    for i in range(1, n):
        res = xNext2(res)
    return res
    
def rxNext1(n):
    res = xNext1(x)
    for i in range(1, n):
        res = xNext1(res)
    return res

def rxNext1v2(n):
    res = xNext1(x)
    for i in range(1, n):
        res = xNext2(res)
    return res

# This output does not match the paper.
td = lambda nval: uprints(rxNext2(nval) - rxNext1(nval))

# This one does.
tdv2 = lambda nval: uprints(rxNext2(nval) - rxNext1v2(nval))