""""Implementations of Move classes.

A Move is used to generate move/step proposals for Monte Carlo sampling.
For example a Flip is simply proposes a change of the identity of a species at
a site, for use in a SemiGrand ensemble. A Swap will propose a swap between
species at two sites for use in Canonical ensemble simulations.

More complex moves can be defined simply by deriving from the BaseMove.
"""

from abc import ABC, abstractmethod
import random


class BaseMove(ABC):
    """Abstract base class for move classes."""

    @abstractmethod
    def propose(self, occupancy, site_space, site_indices):
        """Propose an elementary move.

        A move is given as a sequence of tuples, where each tuple is of the
        form (site idex, species code to set)

        Args:
            occupancy (ndarray):
                encoded occupancy string.
            site_space (OrderedDict):
                An ordered dict with the allowed species and their random
                state composition. See definitions in cofe.cofigspace.basis
            site_indices (ndarray):
                array with the site indices

        Returns:
            list(tuple): tuple of tuples each with (idex, code)
        """
        return []


class Swap(BaseMove):
    """Implementation of a simple swap move."""

    def propose(self, occupancy, site_space, site_indices):
        """Propose a single swap move.

        A move is given as a sequence of tuples, where each tuple is of the
        form (site idex, species code to set)

        Args:
            occupancy (ndarray):
                encoded occupancy string.
            site_space (OrderedDict):
                An ordered dict with the allowed species and their random
                state composition. See definitions in cofe.cofigspace.basis
            site_indices (ndarray):
                array with the site indices

        Returns:
            list(tuple): list of tuples each with (idex, code)
        """

        site1 = random.choice(site_indices)
        swap_options = [i for i in site_indices
                        if occupancy[i] != occupancy[site1]]
        if swap_options:
            site2 = random.choice(swap_options)
            swap = [(site1, occupancy[site2]),
                    (site2, occupancy[site1])]
        else:
            # inefficient, maybe re-call method? infinite recursion problem
            swap = []

        return swap


class Flip(BaseMove):
    """Implementation of a simple flip move."""

    def propose(self, occupancy, site_space, site_indices):
        """Propose a single swap move.

        A move is given as a sequence of tuples, where each tuple is of the
        form (site idex, species code to set)

        Args:
            occupancy (ndarray):
                encoded occupancy string.
            site_space (OrderedDict):
                An ordered dict with the allowed species and their random
                state composition. See definitions in cofe.cofigspace.basis
            site_indices (ndarray):
                array with the site indices

        Returns:
            list(tuple): list of tuples each with (idex, code)
        """

        site = random.choice(site_indices)
        choices = set(range(len(site_space))) - {occupancy[site]}
        return [(site, random.choice(list(choices)))]
