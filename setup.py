"""smol -- Statistical Mechanics On Lattices"""
import os
import platform
import shutil
import sys

import numpy
from setuptools import Command, Extension, setup
from setuptools.command.build_ext import build_ext as build_ext_

from tools.build_helpers import (
    check_openmp_support,
    cythonize_extensions,
    get_openmp_flag,
)

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
    if "PYTHON_CROSSENV" not in os.environ:
        COMPILE_OPTIONS["other"] += ["-mcpu=native"]
    elif platform.machine() == "x86_64":  # cross compiling for macos arm64
        COMPILE_OPTIONS["other"] += ["-mcpu=apple-m1"]
    # otherwise we don't specific -mcpu


# custom clean command to remove .c files
# taken from sklearn setup.py
class clean(Command):
    description = "Remove build artifacts from the source tree"

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Remove c files if we are not within a sdist package
        cwd = os.path.abspath(os.path.dirname(__file__))
        remove_c_files = not os.path.exists(os.path.join(cwd, "PKG-INFO"))
        if remove_c_files:
            print("Will remove generated .c files")
        if os.path.exists("build"):
            shutil.rmtree("build")
        for dirpath, dirnames, filenames in os.walk("smol"):
            for filename in filenames:
                _, extension = os.path.splitext(filename)

                if extension in [".so", ".pyd", ".dll", ".pyc"]:
                    os.unlink(os.path.join(dirpath, filename))

                if remove_c_files and extension in [".c"]:
                    pyx_file = str.replace(filename, extension, ".pyx")
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))

            for dirname in dirnames:
                if dirname == "__pycache__":
                    shutil.rmtree(os.path.join(dirpath, dirname))


class build_ext_options:
    def build_options(self):
        OMP_SUPPORTED = check_openmp_support(self.compiler)
        for e in self.extensions:
            e.extra_compile_args += COMPILE_OPTIONS.get(
                self.compiler.compiler_type, COMPILE_OPTIONS["other"]
            )

            e.extra_link_args += LINK_OPTIONS.get(
                self.compiler.compiler_type, LINK_OPTIONS["other"]
            )
            if OMP_SUPPORTED:
                omp_flag = get_openmp_flag(self.compiler)
                e.extra_compile_args += omp_flag
                e.extra_link_args += omp_flag


class build_ext(build_ext_, build_ext_options):
    def build_extensions(self):
        build_ext_options.build_options(self)
        build_ext_.build_extensions(self)


# Compile option for cython extensions
cython_kwargs = {}
if "--force-cython" in sys.argv:
    cython_kwargs["force"] = True
    sys.argv.remove("--force-cython")
if "--annotate-cython" in sys.argv:
    cython_kwargs["annotate"] = True
    sys.argv.remove("--annotate-cython")

extensions = [
    Extension(
        "smol.utils.cluster.evaluator",
        ["smol/utils/cluster/evaluator.pyx"],
        language="c",
    ),
    Extension(
        "smol.utils.cluster.ewald",
        ["smol/utils/cluster/ewald.pyx"],
        language="c",
    ),
    Extension(
        "smol.utils.cluster.container",
        ["smol/utils/cluster/container.pyx"],
        language="c",
    ),
    Extension(
        "smol.utils.cluster.correlations",
        ["smol/utils/cluster/correlations.pyx"],
        language="c",
    ),
    Extension(
        "smol.utils._openmp_helpers",
        ["smol/utils/_openmp_helpers.pyx"],
        language="c",
    ),
]

cmdclass = {
    "clean": clean,
    "build_ext": build_ext,
}

setup(
    ext_modules=cythonize_extensions(extensions, **cython_kwargs),
    cmdclass=cmdclass,
    include_dirs=numpy.get_include(),
)
