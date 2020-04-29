from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy


# define an extension that will be cythonized and compiled
ext = Extension(name="fastStonks", sources=["fastStonks.pyx"], include_dirs=[numpy.get_include()])
setup(ext_modules=cythonize(ext, compiler_directives={'language_level' : "3"}, annotate=True))
