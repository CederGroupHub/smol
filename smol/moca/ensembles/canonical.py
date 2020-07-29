"""
Implementation of a Canonical Ensemble Class.

Used when running Monte Carlo simulations for fixed number of sites and fixed
concentration of species.
"""

__author__ = "Luis Barroso-Luque"
__credits__ = "Daniil Kitcheav"

import random
from itertools import combinations
from collections import defaultdict
import numpy as np
from monty.json import MSONable
from smol.moca.ensembles.base import BaseEnsemble
from smol.moca.processor import BaseProcessor
from smol.constants import kB


class CanonicalEnsemble(BaseEnsemble, MSONable):
    """
    A Canonical Ensemble class to run Monte Carlo Simulations.

    Attributes:
        temperature (float): temperature in Kelvin

    """

    def __init__(self, processor, temperature, sample_interval,
                 initial_occupancy, sublattices=None, site_space_overlap=False,
                 seed=None):
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
            site_space_overlap (bool): optional
                For systems with sets of 2 or more allowed species in two
                different sublattices (as determined by their associated
                site spaces), setting to true will allow species swaps between
                these sublattices.
            seed (int):
                Seed for random number generator.

        """
        super().__init__(processor, initial_occupancy=initial_occupancy,
                         sample_interval=sample_interval, seed=seed,
                         sublattices=sublattices)
        self.temperature = temperature
        self._min_energy = self._property
        self._min_occupancy = self._init_occupancy

        self._sublattice_overlap = None
        # get sublattice species intersection sets, fill overlap dict
        if site_space_overlap:
            self._sublattice_overlap = defaultdict(list)
            for sublatt1, sublatt2 in combinations(self._sublattices, 2):
                species1 = self._sublattices[sublatt1]['site_space'].keys()
                species2 = self._sublattices[sublatt2]['site_space'].keys()
                overlap = set(species1).intersection(species2)
                if len(overlap) >= 2:
                    for sp in overlap:
                        overlap_info = ((sublatt1, sublatt2), overlap - {sp})
                        self._sublattice_overlap[sp].append(overlap_info)

    @property
    def temperature(self):
        """Get the temperature of ensemble."""
        return self._temperature

    @temperature.setter
    def temperature(self, T):
        """Set the temperature and beta accordingly."""
        self._temperature = T
        self._beta = 1.0/(kB*T)

    @property
    def beta(self):
        """Get 1/kBT."""
        return self._beta

    @property
    def current_energy(self):
        """Get the energy of structure in the last iteration."""
        return self.current_property

    @property
    def average_energy(self):
        """Get the average of sampled energies."""
        return self.energy_samples.mean()

    @property
    def energy_variance(self):
        """Get the variance of samples energies."""
        return self.energy_samples.var()

    @property
    def energy_samples(self):
        """Get an array with the sampled energies."""
        return np.array([d['energy'] for d
                         in self.data[self._prod_start:]])

    @property
    def minimum_energy(self):
        """Get the minimum energy in samples."""
        return self._min_energy

    @property
    def minimum_energy_occupancy(self):
        """Get the occupancy for of the structure with minimum energy."""
        return self.processor.decode_occupancy(self._min_occupancy)

    @property
    def minimum_energy_structure(self):
        """Get the structure with minimum energy in samples."""
        return self.processor.structure_from_occupancy(self._min_occupancy)

    def anneal(self, start_temperature, steps, mc_iterations,
               cool_function=None, set_min_occu=True):
        """Carry out a simulated annealing procedure.

        Uses the total number of temperatures given by "steps" interpolating
        between the start and end temperature according to a cooling function.
        The start temperature is the temperature set for the ensemble.

        Args:
           start_temperature (float):
               Starting temperature. Must be higher than the current ensemble
               temperature.
           steps (int):
               Number of temperatures to run MC simulations between start and
               end temperatures.
           mc_iterations (int):
               number of Monte Carlo iterations to run at each temperature.
           cool_function (str):
               A (monotonically decreasing) function to interpolate
               temperatures.
               If none is given, linear interpolation is used.
           set_min_occu (bool):
               When True, sets the current occupancy and energy of the
               ensemble to the minimum found during annealing.
               Otherwise, do a full reset to initial occupancy and energy.

        Returns:
           tuple: (minimum energy, occupation, annealing data)

        """
        if start_temperature < self.temperature:
            raise ValueError('End temperature is greater than start '
                             f'temperature {self.temperature} > '
                             f'{start_temperature}.')
        if cool_function is None:
            temperatures = np.linspace(start_temperature, self.temperature,
                                       steps)
        else:
            raise NotImplementedError('No other cooling functions implemented '
                                      'yet.')

        min_energy = self.minimum_energy
        min_occupancy = self.minimum_energy_occupancy
        anneal_data = {}

        for T in temperatures:
            self.temperature = T
            self.run(mc_iterations)
            anneal_data[T] = self.data
            self._data = []

        min_energy = self._min_energy
        min_occupancy = self.processor.decode_occupancy(self._min_occupancy)
        self.reset()
        if set_min_occu:
            self._occupancy = self.processor.encode_occupancy(min_occupancy)
            self._property = min_energy

        return min_energy, min_occupancy, anneal_data

    def reset(self):
        """Reset the ensemble by returning it to its initial state.

        This will also clear the all the sample data.
        """
        super().reset()
        self._min_energy = self.processor.compute_property(self._occupancy)
        self._min_occupancy = self._occupancy

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

        sublatt_name = random.choice(sublattices)
        sites = self._active_sublatts[sublatt_name]['sites']
        site1 = random.choice(sites)
        occu1 = self._occupancy[site1]

        swap_options = [i for i in sites if self._occupancy[i] != occu1]
        if self._sublattice_overlap:
            sspace = self._active_sublatts[sublatt_name]['site_space']
            sp = list(sspace.keys())[occu1]
            for (sublatt1, sublatt2), sp_compliment in \
                    self._sublattice_overlap[sp]:
                if self._sublattices[sublatt1]['site_space'] == sspace:
                    swap_sublatt = sublatt2
                elif self._sublattices[sublatt2]['site_space'] == sspace:
                    swap_sublatt = sublatt1
                else:
                    raise RuntimeError('Something has gone real off!!!')
                swap_species = \
                    list(self._sublattices[swap_sublatt]['site_space'].keys())
                allowed_swaps = [swap_species.index(s) for s in sp_compliment]
                swap_options += [i for i in
                                 self._sublattices[swap_sublatt]['sites']
                                 if self._occupancy[i] in allowed_swaps]

        if swap_options:
            site2 = random.choice(swap_options)
            sp = self.processor.allowed_species[site1][self._occupancy[site1]]
            sp2 = self.processor.allowed_species[site2][self._occupancy[site2]]

            return ((site1, self.processor.allowed_species[site1].index(sp2)),
                    (site2, self.processor.allowed_species[site2].index(sp)))
        else:
            # inefficient, maybe re-call method? infinite recursion problem
            return tuple()

    def _get_current_data(self):
        """Get ensemble specific data for current MC step."""
        data = super()._get_current_data()
        data['energy'] = self.current_energy
        return data

    def as_dict(self) -> dict:
        """Json-serialization dict representation.

        Returns:
            MSONable dict

        """
        d = {'@module': self.__class__.__module__,
             '@class': self.__class__.__name__,
             'processor': self.processor.as_dict(),
             'temperature': self.temperature,
             'sample_interval': self.sample_interval,
             'initial_occupancy': self.current_occupancy,
             'seed': self.seed,
             '_min_energy': self.minimum_energy,
             '_min_occupancy': self._min_occupancy.tolist(),
             '_sublattices': self._sublattices,
             '_active_sublatts': self._active_sublatts,
             'restricted_sites': self.restricted_sites,
             'data': self.data,
             '_step': self.current_step,
             '_ssteps': self.accepted_steps,
             '_energy': self.current_energy,
             '_occupancy': self._occupancy.tolist()}
        return d

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
                 initial_occupancy=d['initial_occupancy'],
                 seed=d['seed'])
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
