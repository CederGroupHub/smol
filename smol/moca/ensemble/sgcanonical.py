"""
Implementation of a Semi-Grand Canonical Ensemble Class for running Monte Carlo
simulations for fixed number of sites but variable concentration of species.
"""

import random
import numpy as np
from smol.moca.processor import ClusterExpansionProcessor
from smol.moca.ensemble.canonical import CanonicalEnsemble


class SGCanonicalEnsemble(CanonicalEnsemble):
    """
    A Semi-Grand Canonical Ensemble for Monte Carlo Simulations
    """

    def __init__(self, processor, temperature, chemical_potentials,
                 save_interval, initial_occupancy=None, seed=None):
        """
        Args:
            processor (Processor Class):
                A processor that can compute the change in a property given
                a set of flips.
            temperature (float):
                Temperature of ensemble
            chemical_potentials (dict):
                dictionary with species names and chemical potential
            save_interval (int):
                interval of steps to save the current occupancy and property
            inital_occupancy (array):
                Initial occupancy vector. If none is given then a random one
                will be used.
            seed (int):
                seed for random number generator
        """

        super().__init__(processor, temperature, save_interval,
                         initial_occupancy=initial_occupancy,
                         seed=seed)

        # check that species are valid
        species = [sp for sps in processor.unique_bits for sp in sps]
        for sp in chemical_potentials.keys():
            if sp not in species:
                raise ValueError(f'Species {sp} in provided chemical '
                                 f'potentials is not a specie in the expansion'
                                 f': {species}')

        self.chem_pots = chemical_potentials

    def _get_flips(self, sublattice_name=None):
        """
        Gets a possible semi-grand canonical flip.

        Args:
            sublattice_name (str): optional
                If only considering one sublattice.
        Returns:
            tuple
        """
        if sublattice_name is None:
            sublattice_name = random.choice(list(self._sublattices.keys()))

        sublattice = self._sublattices[sublattice_name]

        site = random.choice(sublattice['sites'])
        old_bit = self._occupancy[site]
        choices = set(range(len(sublattice['bits']))) - old_bit
        new_bit = random.choice(list(choices))
        old_species = sublattice['bits'][old_bit]
        new_species = sublattice['bits'][new_bit]

        return (site, new_bit), (old_species, new_species)


    def _attempt_step(self, sublattice_name=None):
        """
        Attempts flips corresponding to a canonical swap
        Args:
            sublattice_name (str): optional
                If only considering one sublattice.

        Returns: Flip acceptance
            bool
        """

        flip, species = self._get_flips(sublattice_name)
        delta_e = self.processor.compute_property_change(self._occupancy,
                                                         [flip])

        delta_mu = self.chem_pots[species[1]] - self.chem_pots[species[0]]
        delta_phi = delta_e - self.processor.size*delta_mu

        accept = self._accept(delta_phi, self.beta)

        if accept:
            self._energy += delta_e
            self._occupancy[flip[0]] = flip[1]
            if self._energy < self._min_energy:
                self._min_energy = self._energy
                self._min_occupancy = self._occupancy.copy()

        return accept

    def as_dict(self) -> dict:
        """
        Json-serialization dict representation

        Returns:
            MSONable dict
        """
        d = super().as_dict()
        d['chem_pots'] = self.chem_pots
        return d

    @classmethod
    def from_dict(cls, d):
        """
        Creates a CanonicalEnsemble from MSONable dict representation
        """
        eb = cls(ClusterExpansionProcessor.from_dict(d['processor']),
                 temperature=d['temperature'],
                 chemical_potentials=d['chem_pots'],
                 save_interval=d['save_interval'],
                 initial_occupancy=d['initial_occupancy'],
                 seed=d['seed'])
        eb._min_energy = d['_min_energy']
        eb._min_occupancy = np.array(d['_min_occupancy'])
        eb._sublattices = d['_sublattices']
        eb._data = d['_data']
        eb._step = d['_step']
        eb._ssteps = d['_ssteps']
        eb._energy = d['_energy']
        eb._occupancy = np.array(d['_occupancy'])
        return eb
