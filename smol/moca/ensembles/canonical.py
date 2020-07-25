"""
Implementation of a Canonical Ensemble Class.

Used when running Monte Carlo simulations for fixed number of sites and fixed
concentration of species.
"""

__author__ = "Luis Barroso-Luque"
__credits__ = "Daniil Kitcheav"

import random
import numpy as np

from monty.json import MSONable
from smol.moca.ensembles.base import ThermoEnsemble
from smol.moca.processors.base import BaseProcessor


class CanonicalEnsemble(ThermoEnsemble, MSONable):
    """
    A Canonical Ensemble class to run Monte Carlo Simulations.

    Attributes:
        temperature (float): temperature in Kelvin
    """

    def __init__(self, processor, temperature, sample_interval,
                 initial_occupancy, sublattices=None):
        """Initialize CanonicalEnemble.

        Args:
            processor (Processor):
                A processor that can compute the change in a property given
                a set of flips.
            temperature (float):
                Temperature of ensemble
            sample_interval (int):
                Interval of steps to save the current occupancy and energy
            initial_occupancy (ndarray or list):
                Initial occupancy vector. The occupancy can be encoded
                according to the processor or the species names directly.
            sublattices (dict): optional
                dictionary with keys identifying the active sublattices
                (i.e. "anion" or the allowed species in that sublattice
                "Li+/Vacancy". The values should be a dictionary
                with two items {'sites': array with the site indices for all
                sites corresponding to that sublattice in the occupancy vector,
                'site_space': OrderedDict of allowed species in sublattice}
                All sites in a sublattice need to have the same set of allowed
                species.
        """
        super().__init__(processor, temperature=temperature,
                         sample_interval=sample_interval,
                         initial_occupancy=initial_occupancy,
                         sublattices=sublattices)

    def _attempt_step(self, sublattices=None):
        """Attempt flips corresponding to an elementary canonical swap.

        Will pick a sublattice at random and then a canonical swap at random
        from that sublattice (frozen sites will be excluded).

        Args:
            sublattices (list of str): optional
                If only considering one sublattice.

        Returns: Flip acceptance
            bool
        """
        flips = self._get_flips(sublattices)
        delta_e = self.processor.compute_property_change(self._occupancy,
                                                         flips)
        accept = self._accept(delta_e, self.beta)

        if accept:
            self._property += delta_e
            for f in flips:
                self._occupancy[f[0]] = f[1]
            if self._property < self._min_energy:
                self._min_energy = self._property
                self._min_occupancy = self._occupancy.copy()

        return accept

    def _get_flips(self, sublattices=None):
        """Get a possible canonical flip. A swap between two sites.

        Args:
            sublattices (list of str): optional
                If only considering one sublattice.
        Returns:
            tuple
        """
        if sublattices is None:
            sublattices = self.sublattices

        sublattice_name = random.choice(sublattices)
        sites = self._active_sublatts[sublattice_name]['sites']
        site1 = random.choice(sites)
        swap_options = [i for i in sites
                        if self._occupancy[i] != self._occupancy[site1]]
        if swap_options:
            site2 = random.choice(swap_options)
            return ((site1, self._occupancy[site2]),
                    (site2, self._occupancy[site1]))
        else:
            # inefficient, maybe re-call method? infinite recursion problem
            return tuple()

    @classmethod
    def from_dict(cls, d):
        """Create a CanonicalEnsemble from MSONable dict representation.

        Args:
            d (dict): dictionary from CanonicalEnsemble.as_dict()

        Returns:
            CanonicalEnsemble
        """
        eb = cls(BaseProcessor.from_dict(d['processor']),
                 temperature=d['temperature'],
                 sample_interval=d['sample_interval'],
                 initial_occupancy=d['initial_occupancy'])
        eb._min_energy = d['_min_energy']
        eb._min_occupancy = np.array(d['_min_occupancy'])
        eb._sublattices = d['_sublattices']
        eb._active_sublatts = d['_active_sublatts']
        eb.restricted_sites = d['restricted_sites']
        eb._data = d['data']
        eb._step = d['_step']
        eb._ssteps = d['_ssteps']
        eb._property = d['_energy']
        eb._occupancy = np.array(d['_occupancy'])
        return eb
