from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Hello world app',
    ext_modules=cythonize(["./module/image_pad.pyx", "./module/regression.pyx"], language_level = "3", annotate=True),
)

# Install Cython with pip, and build c object file with below command.
# python cython_setup.py build_ext --inplace