from setuptools import setup
from Cython.Build import cythonize

setup(
    name='MyCythonModule',
    ext_modules=cythonize("kron_test.pyx"),
    zip_safe=False,
)