
from setuptools import setup
from Cython.Build import cythonize

# We need to get the directory of Numpy's .h headers using numpy.get_include(). 
# Meanwhile, to subclass sklearn Cython code, it's sufficient to run setup.py
# in an environment with sklearn installed.
import numpy

setup(
    name='weighted_impurity',
    ext_modules=cythonize("weighted_impurity.pyx", annotate=True),
    include_dirs=[numpy.get_include()]
)
