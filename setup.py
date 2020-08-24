# coding: utf-8
# Copyright (c)
# Distributed under the terms of the MIT License.

"""Setup.py for smol."""

import sys
import platform

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import numpy


class build_ext(_build_ext):
    """Extension builder that checks for numpy before install."""

    def finalize_options(self):
        """Override finalize_options."""
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        import builtins
        if hasattr(builtins, '__NUMPY_SETUP__'):
            del builtins.__NUMPY_SETUP__
        import importlib
        importlib.reload(numpy)
        self.include_dirs.append(numpy.get_include())


extra_link_args = []
if sys.platform.startswith('win') and platform.machine().endswith('64'):
    extra_link_args.append('-Wl,--allow-multiple-definition')

cpp_extra_link_args = extra_link_args
cpp_extra_compile_args = ["-Wno-cpp", "-Wno-unused-function", "-O2", "-march=native", '-std=c++11']
if sys.platform.startswith('darwin'):
    cpp_extra_compile_args.append("-stdlib=libc++")
    cpp_extra_link_args = ["-O2", "-march=native", '-stdlib=libc++']

long_desc = """
Lighthweight but caffeinated Python implementations of computational methods
for statistical mechanical calculations of configurational states for
crystalline material systems.
"""

# Compile option for cython extensions
if '--use-cython' in sys.argv:
    USE_CYTHON = True
    cython_kwargs = {}
    sys.argv.remove('--use-cython')
    if '--annotate-cython' in sys.argv:
        cython_kwargs['annotate'] = True
        sys.argv.remove('--annotate-cython')
else:
    USE_CYTHON = False

ext = '.pyx' if USE_CYTHON else '.c'
ext_modules = [Extension("src.mc_utils",
                         ["src/mc_utils"+ext],
                         language="c",
                         include_dirs=["src/"],
                         extra_compile_args=["-O3", "-ffast-math"]
                         )]

if USE_CYTHON:
    from Cython.Build import cythonize
    ext_modules = cythonize(ext_modules,
                            include_path=[numpy.get_include()],
                            compiler_directives={'language_level': 3},
                            **cython_kwargs)

setup(
    name="smol",
    packages=find_packages(),
    version="v1.0.0",
    cmdclass={"build_ext": build_ext},
    setup_requires=['numpy>=1.18.1', 'setuptools>=18.0'],
    python_requires='>=3.7',
    install_requires=['numpy>=1.18.1', 'pymatgen>=2020.8.13',
                      'monty>=3.0.1', 'cvxopt'],
    extras_require={
        "provenance": ["pybtex"],
        ':python_version > "3.7"': [
            "dataclasses>=0.6",
        ]},
    package_data={},
    author="Ceder Group CE Development",
    author_email="lbluque@berkeley.edu",
    maintainer="Who takes this for the team?",
    maintainer_email="above@beep.com",
    url="https://ceder.berkeley.edu/",
    license="MIT",
    description="Computational methods for statistical mechanical and"
                "thermodynamics calculations of crystalline materials systems.",
    long_description=long_desc,
    long_description_content_type='text/markdown',
    keywords=["materials", "science", "structure", "analysis", "phase",
              "diagrams", "crystal", "clusters", "Monte Carlo", "inference"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    ext_modules=ext_modules,
    project_urls={
        'Source': 'https://github.com/CederGroupHub/smol',
        'Bug Reports': 'https://github.com/CederGroupHub/smol/issues'
    },
)