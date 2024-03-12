from setuptools import setup
from Cython.Build import cythonize

setup(
    name='MyCythonModule',
    ext_modules=cythonize("Photoenergy.pyx"),
    zip_safe=False,
)