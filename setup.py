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

import numpy
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

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
    Extension(
        "src.mc_utils", ["src/mc_utils" + ext], language="c", include_dirs=["src/"]
    )
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
    name="smol",
    version="v0.0.0",
    author="Luis Barroso-Luque",
    author_email="lbluque@berkeley.edu",
    maintainer="Luis Barroso-Luque",
    maintainer_email="lbluque@berkeley.edu",
    url="https://ceder.berkeley.edu/",
    license="MIT",
    description="Lighthweight but caffeinated Python implementations of computational"
    " methods for statistical mechanical calculations of configurational states in "
    "crystalline material systems.",
    long_description_content_type="text/markdown",
    keywords=[
        "materials",
        "science",
        "structure",
        "analysis",
        "phase",
        "diagrams",
        "crystal",
        "clusters",
        "Monte Carlo",
        "inference",
    ],
    packages=find_packages(),
    cmdclass={"build_ext": build_ext_subclass},
    setup_requires=["numpy>=1.18.1", "setuptools>=18.0"],
    python_requires=">=3.8",
    tests_require=["pytest"],
    install_requires=[
        "setuptools",
        "numpy>=1.18.1",
        "pymatgen>=2022.0.2",
        "monty>=3.0.1",
    ],
    extras_require={
        "docs": [
            "sphinx==4.5.0",
            "pydata-sphinx-theme==0.8.1",
            "ipython==8.2.0",
            "nbsphinx==0.8.8",
            "nbsphinx-link==1.3.0",
            "nb2plots==0.6.0",
        ],
        "tests": [
            "pytest==7.1.1",
            "pytest-cov==3.0.0",
            "scikit-learn>=0.24.2",
            "h5py>=3.5.0",
        ],
        "dev": ["pre-commit>=2.12.1", "cython>=0.29.24"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    project_urls={
        "Source": "https://github.com/CederGroupHub/smol",
        "Bug Reports": "https://github.com/CederGroupHub/smol/issues",
    },
    ext_modules=ext_modules,
    include_dirs=numpy.get_include(),
)
