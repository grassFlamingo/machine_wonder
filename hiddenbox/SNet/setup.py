from distutils.core import setup
from Cython.Build import cythonize

# setup.py build_ext --inplace

setup(
  name = 'CnnLayers',
  ext_modules = cythonize("CnnLayers.pyx"),
)