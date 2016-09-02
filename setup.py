from distutils.core import setup, Extension
import numpy as np

ext_modules = [ Extension('ml.utils',
                          sources = ['ml/utils.c'],
                          include_dirs=[np.get_include()]) ]

setup(
        name = 'ml',
        version = '0.1',
        include_dirs = [np.get_include()],
        ext_modules = ext_modules,
        packages = ['ml']
    )
