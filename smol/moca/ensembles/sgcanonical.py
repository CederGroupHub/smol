"""
Implementation of a Semi-Grand Canonical Ensemble Class for running Monte Carlo
simulations for fixed number of sites but variable concentration of species.
"""

import random
import numpy as np
from smol.moca.processor import ClusterExpansionProcessor
from smol.moca.ensembles.canonical import CanonicalEnsemble


class SGCanonicalEnsemble(CanonicalEnsemble):
    """
    A Semi-Grand Canonical Ensemble for Monte Carlo Simulations where species
    chemical potentials are predefined. Note that int the SGEnsemble
    implemented in this way, only the differences in chemical potentials with
    respect to a reference species on each sublattice are fixed, and not the
    absolute values. To obtain the absolute values you must calculate the
    reference chemical potential and then simply subtract it from the given
    values.
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
                dictionary with species names and chemical potentials. If the
                chemical potential for one species is not zero (reference), one
                will be chosen and all other values will be shifted accordingly
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

        # Add chemical potentials to sublattice dictionary
        for sublattice in self._sublattices.values():
            sublattice['mu'] = {sp : mu for sp, mu
                                in chemical_potentials.items()
                                if sp in sublattice['bits']}
            # If no reference species is set, then set and recenter others
            mus = list(sublattice['mu'].values())
            if not any([mu == 0 for mu in mus]):
                ref_mu = mus[0]
                sublattice['mu'] = {sp: mu - ref_mu for sp, mu
                                    in sublattice['mu'].items()}

    @property
    def chemical_potentials(self):
        """Relative chemical potentials. Reference species have 0"""
        chem_pots = {}
        for sublattice in self._sublattices.values():
            chem_pots.update(sublattice['mu'])
        return chem_pots

    @property
    def species_counts(self):
        """
        Counts of species. This excludes "static" species. Those with no
        partial occupancy
        """
        counts = self._get_counts()
        species_counts = {}

        for name in self._sublattices.keys():
            cons = {sp: count for sp, count
                    in zip(self._sublattices[name]['bits'], counts[name])}
            species_counts.update(cons)

        return species_counts

    def _get_flips(self, sublattice_name=None):
        """
        Gets a possible semi-grand canonical flip, and the corresponding
        change in chemical potential

        Args:
            sublattice_name (str): optional
                If only considering one sublattice.
        Returns: flip, delta_mu
            tuple
        """
        if sublattice_name is None:
            sublattice_name = random.choice(list(self._sublattices.keys()))

        sublattice = self._sublattices[sublattice_name]

        site = random.choice(sublattice['sites'])
        old_bit = self._occupancy[site]
        choices = set(range(len(sublattice['bits']))) - {old_bit}
        new_bit = random.choice(list(choices))
        old_species = sublattice['bits'][old_bit]
        new_species = sublattice['bits'][new_bit]
        delta_mu = sublattice['mu'][new_species] - sublattice['mu'][old_species]  # noqa

        return (site, new_bit), delta_mu

    def _attempt_step(self, sublattice_name=None):
        """
        Attempts flips corresponding to a canonical swap
        Args:
            sublattice_name (str): optional
                If only considering one sublattice.

        Returns: Flip acceptance
            bool
        """
        flip, delta_mu = self._get_flips(sublattice_name)
        delta_e = self.processor.compute_property_change(self._occupancy,
                                                         [flip])

        delta_phi = delta_e - delta_mu
        accept = self._accept(delta_phi, self.beta)

        if accept:
            self._energy += delta_e
            self._occupancy[flip[0]] = flip[1]
            if self._energy < self._min_energy:
                self._min_energy = self._energy
                self._min_occupancy = self._occupancy.copy()

        return accept

    def _get_counts(self):
        """
        Get the total count of each species for current occupation

        Returns: dict of sublattices with corresponding species concentrations
            dict
        """
        counts = {}
        for name, sublattice in self._sublattices.items():
            occupancy = self._occupancy[sublattice['sites']]
            counts[name] = [np.count_nonzero(occupancy == sp)
                            for sp in range(len(sublattice['bits']))]
        return counts

    def _get_current_data(self):
        """
        Get ensemble specific data for current MC step
        """
        data = super()._get_current_data()
        data['counts'] = self.species_counts

        return data

    def as_dict(self) -> dict:
        """
        Json-serialization dict representation

        Returns:
            MSONable dict
        """
        d = super().as_dict()
        d['chem_pots'] = self.chemical_potentials
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
