"""Utilities for building smol with OpenMP. Borrowed mostly from astropy and sklearn."""

import glob
import os
import subprocess
import sys
import tempfile
import textwrap
import traceback
import warnings

from setuptools.command.build_ext import customize_compiler, new_compiler

OMP_FLAGS = {
    "msvc": ["/openmp"],
    "mingw32": ["-fopenmp"],
    "other": ["-fopenmp"],
}


def compile_test_program(code, extra_preargs=None, extra_postargs=None):
    """Check that some C code can be compiled and run."""
    compiler = new_compiler()
    customize_compiler(compiler)

    start_dir = os.path.abspath(".")

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            os.chdir(tmp_dir)

            # Write test program
            with open("test_program.c", "w") as f:
                f.write(code)

            os.mkdir("objects")
            # Compile, test program
            compiler.compile(
                ["test_program.c"], output_dir="objects", extra_postargs=extra_postargs
            )

            # Link test program
            objects = glob.glob(os.path.join("objects", "*" + compiler.obj_extension))
            compiler.link_executable(
                objects,
                "test_program",
                extra_preargs=extra_preargs,
                extra_postargs=extra_postargs,
            )

            if "PYTHON_CROSSENV" not in os.environ:
                # Run test program if not cross compiling
                # will raise a CalledProcessError if return code was non-zero
                output = subprocess.check_output("./test_program")
                output = output.decode(sys.stdout.encoding or "utf-8").splitlines()
            else:
                # Return an empty output if we are cross compiling
                # as we cannot run the test_program
                output = []
        except Exception:
            raise
        finally:
            os.chdir(start_dir)

    return output


def basic_check_build():
    """Check basic compilation and linking of C code."""
    code = textwrap.dedent(
        """\
        #include <stdio.h>
        int main(void) {
        return 0;
        }
        """
    )
    compile_test_program(code)


def get_openmp_flag(compiler):
    """Find the correct flag to enable OpenMP support."""
    if sys.platform.startswith("darwin") and "openmp" in os.getenv("CPPFLAGS", ""):
        # -fopenmp can't be passed as compile flag when using Apple-clang.
        # OpenMP support has to be enabled during preprocessing.
        #
        # For example, our macOS wheel build jobs use the following environment
        # variables to build with Apple-clang and the brew installed "libomp":
        #
        # export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
        # export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
        # export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
        # export LDFLAGS="$LDFLAGS -Wl,-rpath,/usr/local/opt/libomp/lib
        #                          -L/usr/local/opt/libomp/lib -lomp"
        return []
    else:
        return OMP_FLAGS.get(compiler.compiler_type, OMP_FLAGS["other"])


def check_openmp_support(compiler):
    """Check whether OpenMP test code can be compiled and run."""
    code = textwrap.dedent(
        """\
        #include <omp.h>
        #include <stdio.h>
        int main(void) {
        #pragma omp parallel
        printf("nthreads=%d\\n", omp_get_num_threads());
        return 0;
        }
        """
    )
    extra_preargs = os.getenv("LDFLAGS", None)
    if extra_preargs is not None:
        extra_preargs = extra_preargs.strip().split(" ")
        # FIXME: temporary fix to link against system libraries on linux
        # "-Wl,--sysroot=/" should be removed
        extra_preargs = [
            flag
            for flag in extra_preargs
            if flag.startswith(("-L", "-Wl,-rpath", "-l"))
        ]

    extra_postargs = get_openmp_flag(compiler)

    try:
        output = compile_test_program(
            code, extra_preargs=extra_preargs, extra_postargs=extra_postargs
        )

        if output and "nthreads=" in output[0]:
            nthreads = int(output[0].strip().split("=")[1])
            openmp_supported = len(output) == nthreads
        elif "PYTHON_CROSSENV" in os.environ:
            # Since we can't run the test program when cross-compiling
            # assume that openmp is supported if the program can be
            # compiled.
            openmp_supported = True
        else:
            openmp_supported = False

    except Exception:
        # We could be more specific and only catch: CompileError, LinkError,
        # and subprocess.CalledProcessError.
        # setuptools introduced CompileError and LinkError, but that requires
        # version 61.1. Even the latest version of Ubuntu (22.04LTS) only
        # ships with 59.6. So for now we catch all exceptions and reraise a
        # generic exception with the original error message instead:
        openmp_supported = False
        traceback.print_exc()

    if not openmp_supported:
        message = textwrap.dedent(
            """

                            ***********
                            * WARNING *
                            ***********

            It seems that smol cannot be built with OpenMP.

            - See exception traceback for more details.

            - Make sure you have followed the installation instructions:

                https://cedergrouphub.github.io/smol/getting_started.html

            - If your compiler supports OpenMP but you still see this
              message, please submit a bug report at:

                https://github.com/CederGroupHub/smol/issues

            - The build will continue with OpenMP-based parallelism
              disabled. Note however that computing correlation and cluster interaction
              vectors will run in sequential mode instead of leveraging thread-based
              parallelism.

                                ***\n
            """
        )
        warnings.warn(message)

    return openmp_supported


def cythonize_extensions(extensions, **cythonize_kwargs):
    """Cythonize extensions."""
    import numpy
    from Cython.Build import cythonize

    # Fast fail before cythonization if compiler fails compiling basic test
    # code even without OpenMP
    basic_check_build()

    compiler_directives = {
        "language_level": 3,
        "boundscheck": False,  # set to True for debugging
        "nonecheck": False,
        "wraparound": False,
        "initializedcheck": False,
        "cdivision": True,
        **cythonize_kwargs,
    }

    return cythonize(
        extensions,
        include_path=[numpy.get_include()],
        compiler_directives=compiler_directives,
    )
