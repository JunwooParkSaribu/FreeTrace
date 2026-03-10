# Modified by Claude (claude-opus-4-6, Anthropic AI) — build Cython extensions at pip install time
import os
import numpy as np
from setuptools import setup, Extension

# Try Cython; fall back to pre-generated .c files
try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

_cython_dir = os.path.join("FreeTrace", "module", "cython_build")
_module_dir = os.path.join("FreeTrace", "module")

_modules = ["image_pad", "regression", "cost_function"]

if USE_CYTHON:
    ext_modules = cythonize(
        [
            Extension(
                f"FreeTrace.module.{mod}",
                sources=[os.path.join(_cython_dir, f"{mod}.pyx")],
                include_dirs=[np.get_include()],
            )
            for mod in _modules
        ],
        language_level="3",
    )
else:
    # Fall back to pre-generated C files
    ext_modules = [
        Extension(
            f"FreeTrace.module.{mod}",
            sources=[os.path.join(_module_dir, f"{mod}.c")],
            include_dirs=[np.get_include()],
        )
        for mod in _modules
    ]

setup(ext_modules=ext_modules)
