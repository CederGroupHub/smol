# Copyright (c)

"""
smol -- Statistical Mechanics On Lattices

Lighthweight but caffeinated Python implementations of computational methods
for statistical mechanical calculations of configurational states in
crystalline material systems.


smol is a minimal implementation of computational methods to calculate statistical
mechanical and thermodynamic properties of crystalline material systems based on
the cluster expansion method from alloy theory and related methods. Although smol
is intentionally lightweight---in terms of dependencies and built-in functionality
---it has a modular design that closely follows underlying mathematical formalism
and provides useful abstractions to easily extend existing methods or implement and
test new ones. Finally, although conceived mainly for method development, smol can
(and is being) used in production for materials science research applications.
"""

import sys

from setuptools import Extension, dist, setup
from setuptools.command.build_ext import build_ext

# get numpy to include headers
dist.Distribution().fetch_build_eggs(["numpy>=1.20"])
import numpy

COMPILE_OPTIONS = {
    "msvc": [
        "/O2",
        "/EHsc",
    ],
    "mingw32": ["-Wall", "-Wextra"],
    "other": ["-Wall", "-Wextra", "-ffast-math", "-O3"],
}

LINK_OPTIONS = {
    "msvc": ["-Wl, --allow-multiple-definition"],
    "mingw32": [["-Wl, --allow-multiple-definition"]],
    "other": [],
}
COMPILER_DIRECTIVES = {
    "language_level": 3,
}


if sys.platform.startswith("darwin"):
    COMPILE_OPTIONS["other"] += ["-march=native", "-stdlib=libc++"]


# By subclassing build_extensions we have the actual compiler that will be used which is
# really known only after finalize_options
# http://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used  # noqa
class build_ext_options:
    def build_options(self):
        for e in self.extensions:
            e.extra_compile_args += COMPILE_OPTIONS.get(
                self.compiler.compiler_type, COMPILE_OPTIONS["other"]
            )
        for e in self.extensions:
            e.extra_link_args += LINK_OPTIONS.get(
                self.compiler.compiler_type, LINK_OPTIONS["other"]
            )


class build_ext_subclass(build_ext, build_ext_options):
    def build_extensions(self):
        build_ext_options.build_options(self)
        build_ext.build_extensions(self)


# Compile option for cython extensions
if "--use-cython" in sys.argv:
    USE_CYTHON = True
    cython_kwargs = {}
    sys.argv.remove("--use-cython")
    if "--annotate-cython" in sys.argv:
        cython_kwargs["annotate"] = True
        sys.argv.remove("--annotate-cython")
else:
    USE_CYTHON = False

ext = ".pyx" if USE_CYTHON else ".c"
ext_modules = [
    Extension("smol.correlations", ["smol/correlations" + ext], language="c")
]

if USE_CYTHON:
    from Cython.Build import cythonize

    ext_modules = cythonize(
        ext_modules,
        include_path=[numpy.get_include()],
        compiler_directives={"language_level": 3},
        **cython_kwargs
    )


setup(
    use_scm_version={"version_scheme": "python-simplified-semver"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext_subclass},
    include_dirs=numpy.get_include(),
)
