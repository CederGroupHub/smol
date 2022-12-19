"""Implementation of MCMC transition kernel classes.

A kernel essentially is an implementation of the MCMC algorithm that is used
to generate states for sampling an MCMC chain.
"""

from smol._utils import class_name_from_str, derived_class_factory
from smol.moca.kernel._base import MCKernelInterface
from smol.moca.kernel.metropolis import Metropolis, UniformlyRandom
from smol.moca.kernel.wanglandau import WangLandau

__all__ = ["Metropolis", "UniformlyRandom", "WangLandau"]


def mckernel_factory(kernel_type, ensemble, step_type, *args, **kwargs):
    """Get a MCMC Kernel from string name.

    Args:
        kernel_type (str):
            string specifying kernel type to instantiate.
        ensemble (Ensemble)
            an Ensemble object to create the MCMC kernel from.
        step_type (str):
            string specifying the proposal type (i.e. key for MCUsher type)
        *args:
            positional arguments passed to class constructor
        **kwargs:
            keyword arguments passed to class constructor

    Returns:
        MCKernel: instance of derived class.
    """
    kernel_name = class_name_from_str(kernel_type)
    return derived_class_factory(
        kernel_name, MCKernelInterface, ensemble, step_type, *args, **kwargs
    )
