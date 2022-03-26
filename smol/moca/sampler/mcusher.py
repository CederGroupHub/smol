"""Implementations of MCMC Usher classes.

An usher is used to generate step proposals for MC Monte Carlo sampling.
For example a Flipper simply proposes a change of the identity of a species
at a site, for use in a SemiGrand ensemble. A Swapper will propose a swap
between species at two sites for use in Canonical ensemble simulations.

More complex steps can be defined simply by deriving from the MCUsher
"""

__author__ = "Luis Barroso-Luque"

from abc import ABC, abstractmethod

import numpy as np

from smol.utils import class_name_from_str, derived_class_factory

rng = np.random.default_rng()

class MCUsher(ABC):
    """Abstract base class for MC usher classes."""

    def __init__(
        self, sublattices, inactive_sublattices, sublattice_probabilities=None
    ):
        """Initialize MCMCStep.

        Args:
            sublattices (list of Sublattice):
                list of active Sublattices to propose steps for. Active
                sublattices are those that include sites with configuration
                DOFs.
            inactive_sublattices (list of InactiveSublattice):
                list of inactive Sublattices, i.e. those with no configuration
                DOFs. These can be used to obtain auxiliary information for MC
                step proposals for the active sublattices
            sublattice_probabilities (list of float): optional
                list of probability to pick a site from a specific active
                sublattice.
        """
        self.sublattices = sublattices
        self.inactive_sublattices = inactive_sublattices

        if sublattice_probabilities is None:
            self._sublatt_probs = np.array(
                len(self.sublattices)
                * [
                    1 / len(self.sublattices),
                ]
            )
        elif len(sublattice_probabilities) != len(self.sublattices):
            raise AttributeError(
                "Sublattice probabilites needs to be the " "same length as sublattices."
            )
        elif sum(sublattice_probabilities) != 1:
            raise ValueError("Sublattice probabilites must sum to one.")
        else:
            self._sublatt_probs = sublattice_probabilities

    @property
    def sublattice_probabilities(self):
        """Get the sublattice probabilities."""
        return self._sublatt_probs

    @sublattice_probabilities.setter
    def sublattice_probabilities(self, value):
        """Set the sublattice probabilities."""
        if len(value) != len(self.sublattices):
            raise AttributeError(
                f"Can not set sublattice probabilities.\n Length must be the"
                f" same as the number of sublattices {len(self.sublattices)}"
            )
        if sum(value) != 1:
            raise ValueError(
                "Can not set sublattice probabilities.\n"
                "Sublattice probabilites must sum to one."
            )
        self._sublatt_probs = value

    @abstractmethod
    def propose_step(self, occupancy):
        """Propose an MCMC step.

        A step is given as a sequence of tuples, where each tuple is of the
        form (site index, species code to set)

        Args:
            occupancy (ndarray):
                encoded occupancy string.

        Returns:
            list(tuple): tuple of tuples each with (idex, code)
        """
        return []

    def update_aux_state(self, step, *args, **kwargs):
        """Update any auxiliary state information based on an accepted step."""

    def set_aux_state(self, state, *args, **kwargs):
        """Set the auxiliary state from a checkpoint values."""

    def get_random_sublattice(self):
        """Return a random sublattice based on given probabilities."""
        return rng.choice(a=self.sublattices, p=self._sublatt_probs)


class Flip(MCUsher):
    """Implementation of a simple flip step at a random site."""

    def propose_step(self, occupancy):
        """Propose a single random flip step.

        A step is given as a sequence of tuples, where each tuple is of the
        form (site index, species code to set)

        Args:
            occupancy (ndarray):
                encoded occupancy string.

        Returns:
            list(tuple): list of tuples each with (index, code)
        """
        sublattice = self.get_random_sublattice()
        site = rng.choice(sublattice.active_sites)
        choices = set(range(len(sublattice.site_space))) - {occupancy[site]}
        return [(site, rng.choice(list(choices)))]


class Swap(MCUsher):
    """Implementation of a simple swap step for two random sites."""

    def propose_step(self, occupancy):
        """Propose a single random swap step.

        A step is given as a sequence of tuples, where each tuple is of the
        form (site index, species code to set)

        Args:
            occupancy (ndarray):
                encoded occupancy string.

        Returns:
            list(tuple): list of tuples each with (idex, code)
        """
        sublattice = self.get_random_sublattice()
        site1 = rng.choice(sublattice.active_sites)
        species1 = occupancy[site1]
        sublattice_occu = occupancy[sublattice.active_sites]
        swap_options = sublattice.active_sites[sublattice_occu != species1]
        if swap_options.size > 0:  # check if swap_options are not empty
            site2 = rng.choice(swap_options)
            swap = [(site1, occupancy[site2]), (site2, species1)]
        else:
            # inefficient, maybe re-call method? infinite recursion problem
            swap = []
        return swap


def mcusher_factory(usher_type, sublattices, inactive_sublattices, *args, **kwargs):
    """Get a MC Usher from string name.

    Args:
        usher_type (str):
            string specifying step to instantiate.
        sublattices (list of Sublattice):
                list of Sublattices to propose steps for.
        inactive_sublattices (list of InactiveSublattice):
                list of InactiveSublattices for sites with no configuration
                DOFs.
        *args:
            positional arguments passed to class constructor
        **kwargs:
            Keyword arguments passed to class constructor

    Returns:
        MCUsher: instance of derived class.
    """
    usher_name = class_name_from_str(usher_type)
    return derived_class_factory(
        usher_name, MCUsher, sublattices, inactive_sublattices, *args, **kwargs
    )
