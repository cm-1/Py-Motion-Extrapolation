
from setuptools import setup
from Cython.Build import cythonize

# We need to get the directory of Numpy's .h headers using numpy.get_include(). 
import numpy

setup(
    name='weighted_impurity',
    ext_modules=cythonize("weighted_impurity.pyx"),
    include_dirs=[numpy.get_include()]
)