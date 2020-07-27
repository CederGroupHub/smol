""""Implementations of Move classes.

A Move is used to generate move/step proposals for Monte Carlo sampling.
For example a Flip is simply proposes a change of the identity of a species at
a site, for use in a SemiGrand ensemble. A Swap will propose a swap between
species at two sites for use in Canonical ensemble simulations.

More complex moves can be defined simply by deriving from the BaseMove.
"""

__author__ = "Luis Barroso-Luque"

from abc import ABC, abstractmethod
import random

from smol.utils import derived_class_factory


class MCMCMove(ABC):
    """Abstract base class for move classes."""

    def __init__(self, sublattices, sublattice_probabilities=None):
        """Initialize MCMCMove

        Args:
            sublattices (list of Sublattice):
                list of Sublattices to propose moves for.
            sublattice_probabilities (list of float): optional
                list of probability to pick a site from a specific sublattice.
        """
        self.sublattices = sublattices
        if sublattice_probabilities is None:
            self._sublatt_probs = len(self.sublattices) * [1 / len(self.sublattices), ]  # noqa
        elif len(sublattice_probabilities) != len(self.sublattices):
            raise AttributeError('Sublattice probabilites needs to be the '
                                 'same length as sublattices.')
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
            raise AttributeError('Can not set sublattice probabilities. '
                                 'Length must be the the same as the number '
                                  f'of sublattices {len(self.sublattices)}')
        elif sum(value) != 1:
            raise ValueError('Can not set sublattice probabilities. '
                             'Sublattice probabilites must sum to one.')
        self._sublatt_probs = value

    @abstractmethod
    def propose(self, occupancy):
        """Propose an elementary move.

        A move is given as a sequence of tuples, where each tuple is of the
        form (site idex, species code to set)

        Args:
            occupancy (ndarray):
                encoded occupancy string.

        Returns:
            list(tuple): tuple of tuples each with (idex, code)
        """
        return []

    def get_random_sublattice(self):
        """Return a random sublattice based on given probabilities."""
        return random.choices(self.sublattices, weights=self._sublatt_probs)[0]


class Swap(MCMCMove):
    """Implementation of a simple swap move."""

    def propose(self, occupancy):
        """Propose a single swap move.

        A move is given as a sequence of tuples, where each tuple is of the
        form (site index, species code to set)

        Args:
            occupancy (ndarray):
                encoded occupancy string.

        Returns:
            list(tuple): list of tuples each with (idex, code)
        """
        sublattice = self.get_random_sublattice()
        site1 = random.choice(sublattice.active_sites)
        swap_options = [i for i in sublattice.active_sites
                        if occupancy[i] != occupancy[site1]]
        if swap_options:
            site2 = random.choice(swap_options)
            swap = [(site1, occupancy[site2]),
                    (site2, occupancy[site1])]
        else:
            # inefficient, maybe re-call method? infinite recursion problem
            swap = []

        return swap


class Flip(MCMCMove):
    """Implementation of a simple flip move."""

    def propose(self, occupancy):
        """Propose a single swap move.

        A move is given as a sequence of tuples, where each tuple is of the
        form (site index, species code to set)

        Args:
            occupancy (ndarray):
                encoded occupancy string.

        Returns:
            list(tuple): list of tuples each with (idex, code)
        """
        sublattice = self.get_random_sublattice()
        site = random.choice(sublattice.active_sites)
        choices = set(range(len(sublattice.species))) - {occupancy[site]}
        return [(site, random.choice(list(choices)))]


def mcmc_move_factory(move_type, sublattices, *args, **kwargs):
    """"Get a MCMC move from string name.

    Args:
        move_type (str):
            string specifying move to instantiate.
        sublattices (list of Sublattice):
                list of Sublattices to propose moves for.
        *args:
            positional arguments passed to class constructor
        **kwargs:
            Keyword arguments passed to class constructor

    Returns:
        MCMCMove: instance of derived class.
    """
    return derived_class_factory(move_type.capitalize(), MCMCMove, sublattices)
