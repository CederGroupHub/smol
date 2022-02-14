"""Bias term definitions for biased sampling techniques.

Bias terms can be added to an MCKernel in order to generate samples that are
biased accordingly.
"""

__author__ = "Fengyu Xie, Luis Barroso-Luque"

from abc import ABC, abstractmethod
import numpy as np
from smol.utils import derived_class_factory, class_name_from_str


class MCBias(ABC):
    """Base bias term class.

    Attributes:
        sublattices (List[Sublattice]):
            list of sublattices with active sites.
        inactive_sublattices (List[InactiveSublattice]):
            list of inactive sublattices.
    """

    def __init__(self, sublattices, inactive_sublattices, *args, **kwargs):
        """Initialize Basebias.

        Args:
            sublattices (List[Sublattice]):
                List of active sublattices, containing species information and
                site indices in sublattice.
            inactive_sublattices (List[InactiveSublattice]):
                List of inactive sublattices
            args:
                Additional arguments buffer.
            kwargs:
                Additional keyword arguments buffer.
        """
        self.sublattices = sublattices
        self.inactive_sublattices = inactive_sublattices

    @abstractmethod
    def compute_bias(self, occupancy):
        """Compute bias from occupancy.

        Args:
            occupancy(np.ndarray):
                Encoded occupancy string.
        Returns:
            Float, bias value.
        """
        return

    @abstractmethod
    def compute_bias_change(self, occupancy, step):
        """Compute bias change from step.

        Args:
            occupancy: (ndarray):
                Encoded occupancy array.
            step: (List[tuple(int,int)]):
                Step returned by MCUsher.
        Return:
            Float, change of bias value after step.
        """
        return


def mcbias_factory(bias_type, sublattices, inactive_sublattices, *args,
                   **kwargs):
    """Get a MCMC bias from string name.

    Args:
        bias_type (str):
            string specyting bias name to instantiate.
        sublattices (List[Sublattice]):
            list of active sublattices, containing species information and
            site indices in sublattice.
        inactive_sublattices (List[InactiveSublattice]):
            list of inactive sublattices
        *args:
            positional args to instatiate a bias term.
        *kwargs:
            Keyword argument to instantiate a bias term.
    """
    bias_name = class_name_from_str(bias_type)
    return derived_class_factory(
        bias_name, MCBias, sublattices, inactive_sublattices, *args, **kwargs)
