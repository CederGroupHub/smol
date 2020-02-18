"""
Implementation of Canonical Ensemble Class for running Monte Carlo simulations
for fixed concentration of species.
"""

import random
from monty.json import MSONable
from smol.moca.ensemble.base import BaseEnsemble
from smol.globals import kB


class CanonicalEnsemble(BaseEnsemble, MSONable):
    """
    A Canonical Ensemble class to run Monte Carlo Simulations
    """

    def __init__(self, processor, save_interval, temperature,
                 initial_occupancy=None, seed=None):
        """
        Args:
            processor (Processor Class):
                A processor that can compute the change in a property given
                a set of flips.
            inital_occupancy (array):
                Initial occupancy vector. If none is given then a random one
                will be used.
            save_interval (int):
                interval of steps to save the current occupancy and property
            seed (int):
                seed for random number generator
        """

        super().__init__(processor, initial_occupancy=initial_occupancy,
                         save_interval=save_interval, seed=seed)
        self.temperature = temperature
        self.beta = 1.0/(kB*temperature)
        self._min_energy = self._energy
        self._min_occupancy = self._init_occupancy

    @property
    def minimum_energy(self):
        return self._min_energy

    @property
    def minimum_energy_occupancy(self):
        return self.processor.decode_occupancy(self._min_occupancy)

    @property
    def minimum_energy_structure(self):
        return self.processor.structure_from_occupancy(self._min_occupancy)

    def _attempt_flip(self, flips):
        """
        Attempts flips corresponding to a canonical swap
        Args:
            flips (list):
                list with two tuples consisting of a site index and the encoded
                specie to place at that site.

        Returns: Flip acceptance
            bool
        """
        delta_e = self.processor.compute_property_change(self._occupancy,
                                                         flips)
        accept = self._accept(delta_e, self.beta)

        if accept:
            self._energy += delta_e
            for f in flips:
                self._occupancy[f[0]] = f[1]
            if self._energy < self._min_energy:
                self._min_energy = self._energy
                self._min_occupancy = self._occupancy.copy()

        return accept

    def _get_flip(self, sublattice_name=None):
        """
        Gets a possible canonical flip. A swap between two sites
        Args:
            sublattice_name (str): optional
                If only considering one sublattice.
        Returns:
            tuple
        """
        if sublattice_name is None:
            sublattice_name = random.choice(list(self._sublattices.keys()))

        ind1 = random.choice(self._sublattices[sublattice_name])
        swap_options = [i for i in self._sublattices[sublattice_name]
                        if self._occupancy[i] != self._occupancy[ind1]]
        if swap_options:
            ind2 = random.choice(swap_options)
            return ((ind1, self._occupancy[ind2]),
                    (ind2, self._occupancy[ind1]))
        else:
            # inefficient, maybe re-call method?
            return tuple()

    def _get_current_data(self):
        return {'energy': self.energy, 'occupancy': self.occupancy}

    def as_dict(self) -> dict:
        pass

    def from_dict(cls, d):
        pass
