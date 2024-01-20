

run = True


if run:
    import Photoangular




if not run:

    from setuptools import setup
    from Cython.Build import cythonize

    setup(
        name='MyCythonModule',
        ext_modules=cythonize("Photoangular.pyx"),
        zip_safe=False,
    )