from setuptools import setup
from Cython.Build import cythonize
from petsc4py import get_include

setup(
    ext_modules=cythonize("Module.pyx", annotate=False),
    include_dirs=[get_include()],
)