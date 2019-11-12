 
# coding: utf-8
# Copyright (c)
# Distributed under the terms of the MIT License.

"""Setup.py for SMoL"""

import sys
import platform

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize
import numpy  # this can be put in a function like build_ext to check import and then cythonize

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
        import numpy
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
A long description of this package
"""

setup(
    name="smol",
    packages=find_packages(),
    version="2019.10.4",
    cmdclass={'build_ext': build_ext},
    setup_requires=['numpy>=1.14.3', 'setuptools>=18.0'],
    python_requires='>=3.6',
    install_requires=["numpy>=1.14.3"],
    extras_require={
        "provenance": ["pybtex"],
        "ase": ["ase>=3.3"],
        "vis": ["vtk>=6.0.0"],
        "abinit": ["apscheduler", "netcdf4"],
        ':python_version < "3.7"': [
            "dataclasses>=0.6",
        ]},
    package_data={},
    author="Ceder Group CE Development",
    author_email="lbluque@berkeley.edu",
    maintainer="Who takes this for the team?",
    maintainer_email="above@beep.com",
    #url="",
    license="MIT",
    description="Computational methods for statistical mechanical and thermodynamics"
    			"calculations of crystalline materials systems.",
    long_description=long_desc,
    long_description_content_type='text/markdown',
    keywords=["materials", "science","structure", "analysis", "phase", "diagrams",
    		  "crystal", "clusters", "montecarlo", "inference"],  
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    ext_modules=cythonize("src/ce_utils.pyx", include_path=[numpy.get_include()])
)