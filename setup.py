#!/usr/bin/python

# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

#### Control Flags for Compilation ####

# Optimize runtime binary. Disables output needed for debugging. Implies that
# ndebug = False.
optimize = True

# Disable the NDEBUG preprocessor define, adding (expensive) sanity checks to
# resulting binary.
debug = (not optimize)

# Enable SSE intrinsics.
sse = True

# Add output required for profiling. This generally does not affect the speed of
# the resulting binary.
profiler = False


#### Parsing of Control Flags ####

extra_compile_args = []
extra_link_args = []
define_macros = []
undef_macros = []

if optimize:
  extra_compile_args += [ "-O3", "-mtune=native" ]
  extra_link_args += [ ]
else:
  extra_compile_args += ["-O0", "-g" ]
  extra_link_args += [ "-rdynamic", "-fno-omit-frame-pointer", "-g" ]

if debug:
  undef_macros += [ "NDEBUG" ]
else:
  define_macros += [ ("NDEBUG", None) ]

if sse:
  extra_compile_args += [ "-msse3", "-mfpmath=sse" ]
  extra_link_args += [ "-msse3" ]

if profiler:
  extra_compile_args += [ "-pg" ]
  extra_link_args += [ "-lprofiler", "-pg" ]


#### Do Not Edit Below This Line ####

ext = Extension(
  "glimpse.core.c_src",
  [
    "glimpse/core/c_src.pyx",
    "glimpse/core/c_src/array.cpp",
    "glimpse/core/c_src/bitset_array.cpp",
    "glimpse/core/c_src/retinal_layer.cpp",
    "glimpse/core/c_src/simple_layer.cpp",
    "glimpse/core/c_src/complex_layer.cpp",
    "glimpse/core/c_src/util.cpp",
  ],
  depends = [
    "glimpse/core/c_src/array.h",
    "glimpse/core/c_src/bitset_array.h",
    "glimpse/core/c_src/retinal_layer.h",
    "glimpse/core/c_src/simple_layer.h",
    "glimpse/core/c_src/complex_layer.h",
    "glimpse/core/c_src/util.h",
  ],
  language = "c++",
  extra_compile_args = extra_compile_args,
  extra_link_args = extra_link_args,
  define_macros = define_macros,
  undef_macros = undef_macros,
)

setup(
  name = "glimpse",
  version = "1.0",
  description = "Library for hierarchical visual models in C++ and Python",
  author = "Mick Thomure",
  author_email = "thomure@cs.pdx.edu",
  cmdclass = {'build_ext': build_ext},

  ext_modules = [ ext ],
  packages = [ 'glimpse', 'glimpse.core', 'glimpse.util' ],
  include_dirs = [ numpy.get_include() ],
)