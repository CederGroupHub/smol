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
from smol.moca.ensembles.base import BaseEnsemble
from smol.moca.processor import BaseProcessor
from smol.constants import kB
import time
import itertools
import warnings

Mn_flip_table = {('Mn2+', 'Mn2+'): ['None'],
                 ('Mn2+', 'Mn3+'): ['swap'],
                 ('Mn3+', 'Mn2+'): ['swap'],
                 ('Mn2+', 'Mn4+'): ['dispropA', 'swap'],
                 ('Mn4+', 'Mn2+'): ['dispropA', 'swap'],
                 ('Mn3+', 'Mn3+'): ['dispropB', 'dispropC'],
                 ('Mn3+', 'Mn4+'): ['swap'],
                 ('Mn4+', 'Mn3+'): ['swap'],
                 ('Mn4+', 'Mn4+'): ['None']}


class CanonicalEnsemble(BaseEnsemble, MSONable):
    """
    A Canonical Ensemble class to run Monte Carlo Simulations.

    Attributes:
        temperature (float): temperature in Kelvin
    """

    def __init__(self, processor, temperature, sample_interval,
                 initial_occupancy, sublattices=None, seed=None):
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
            seed (int):
                Seed for random number generator.
        """
        super().__init__(processor, initial_occupancy=initial_occupancy,
                         sample_interval=sample_interval, seed=seed,
                         sublattices=sublattices)
        self.temperature = temperature
        self._min_energy = self._property
        self._min_occupancy = self._init_occupancy

        self._reset_site_table()
        self._sites_to_sublattice = {}
        for sublatt in self._sublattices:
            for site in self._sublattices[sublatt]['sites']:
                self._sites_to_sublattice[site] = sublatt
        self.swap_table = None

        #self.accepted_flip_table = {}
        #self.proposed_flip_table = {}
        #possible_sp = []
        #for site_space in self.processor.unique_site_spaces:
        #    possible_sp += site_space.keys()
        #possible_sp = list(set(possible_sp))
        #sp_sublatt_pairs = []
        #for sp in possible_sp:
        #    sp_str = str(sp)
        #    if 'O' in sp_str or 'F' in sp_str:
        #        sp_sublatt_pairs.append((sp_str, 'anion'))
        #    else:
        #        sp_sublatt_pairs.append((sp_str, 'oct'))
        #        sp_sublatt_pairs.append((sp_str, 'tet'))
        #allowed_swaps = [tuple(sorted([pair1, pair2]))
        #                 for pair1, pair2 in itertools.combinations(sp_sublatt_pairs, 2)]
        #for swap in allowed_swaps:
        #    self.accepted_flip_table[swap] = 0
        #    self.proposed_flip_table[swap] = 0

        #self.flip_type_indices = {}
        #for i, flip_type in enumerate(self.accepted_flip_table):
        #    self.flip_type_indices[flip_type] = i
        #self.acceptance_tracker = []

        # check whether site is oct/tet
        #self._oct_tet_indicator = []
        #for site in self.processor.structure:
        #    neighbors = self.processor.structure.get_neighbors(site, 2.2)
        #    num_neighbors = len([n for n in neighbors if 'O' in n.species_string])
        #    if num_neighbors == 4:
        #        self._oct_tet_indicator.append('tet')
        #    elif num_neighbors == 6:
        #        self._oct_tet_indicator.append('oct')
        #    elif num_neighbors == 0:
        #        self._oct_tet_indicator.append('anion')
        #    else:
        #        print("Calculated incorrect coordination")


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
        self._reset_site_table()

    def restrict_sites(self, sites):
        """Restricts (freezes) the given sites.
        This will exclude those sites from being flipped during a Monte Carlo
        run. If some of the given indices refer to inactive sites, there will
        be no effect.
        Args:
            sites (Sequence):
                indices of sites in the occupancy string to restrict.
        """
        super().restrict_sites(sites)
        self._reset_site_table()

    def _attempt_step(self, sublattices=None, new_method=False):
        """Attempt flips corresponding to an elementary canonical swap.

        Will pick a sublattice at random and then a canonical swap at random
        from that sublattice (frozen sites will be excluded).

        Args:
            sublattices (list of str): optional
                If only considering one sublattice.

        Returns: Flip acceptance
            bool
        """

        if new_method:
            #flips, flip_metadata = self._get_swaps_from_table()
            flips = self._get_swaps_from_table()
        else:
            #flips, flip_metadata  = self._get_flips(sublattices)
            flips  = self._get_flips(sublattices)
        #if flip_metadata:
        #    sp1, sp2, site1, site2 = flip_metadata
        #    site1_type = self._oct_tet_indicator[site1]
        #    site2_type = self._oct_tet_indicator[site2]
        #    flip_type = tuple(sorted([(sp1, site1_type), (sp2, site2_type)]))
        #    self.proposed_flip_table[flip_type] += 1
        delta_e = self.processor.compute_property_change(self._occupancy,
                                                         flips)

        accept = self._accept(delta_e, self.beta)

        if accept:
            self._property += delta_e
            for f in flips:
                self._occupancy[f[0]] = f[1]
                self._update_site_table(flips)
            if self._property < self._min_energy:
                self._min_energy = self._property
                self._min_occupancy = self._occupancy.copy()
            # update accepted flip data
            #if flip_metadata:
            #    self.accepted_flip_table[flip_type] += 1
            #    self.acceptance_tracker.append((self.current_step, self.flip_type_indices[flip_type]))

        return accept

    def _attempt_step_new(self, sublattices=None, new_method=False):
        """Attempt flips corresponding to an elementary canonical swap.

        Will pick a sublattice at random and then a canonical swap at random
        from that sublattice (frozen sites will be excluded).

        Args:
            sublattices (list of str): optional
                If only considering one sublattice.

        Returns: Flip acceptance
            bool
        """

        flips = self._get_swaps_from_table()

        #if flip_metadata:
        #    sp1, sp2, site1, site2 = flip_metadata
        #    site1_type = self._oct_tet_indicator[site1]
        #    site2_type = self._oct_tet_indicator[site2]
        #    flip_type = tuple(sorted([(sp1, site1_type), (sp2, site2_type)]))
        #    self.proposed_flip_table[flip_type] += 1
        delta_e = self.processor.compute_property_change(self._occupancy,
                                                         flips)

        accept = self._accept(delta_e, self.beta)

        if accept:
            self._property += delta_e
            for f in flips:
                self._occupancy[f[0]] = f[1]
                self._update_site_table(flips)
            if self._property < self._min_energy:
                self._min_energy = self._property
                self._min_occupancy = self._occupancy.copy()
            # update accepted flip data
            #if flip_metadata:
            #    self.accepted_flip_table[flip_type] += 1
            #    self.acceptance_tracker.append((self.current_step, self.flip_type_indices[flip_type]))

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
        sp1 = self.processor.allowed_species[site1][self._occupancy[site1]]

        # Build swap_options more quickly from self._site_table
        swap_options = []
        for sp in self._site_table:
            if sp != sp1 and sublattice_name in self._site_table[sp]:
                swap_options += self._site_table[sp][sublattice_name]

        if swap_options:
            site2 = random.choice(swap_options)
            sp2 = self.processor.allowed_species[site2][self._occupancy[site2]]

            return ((site1, self._occupancy[site2]),
                    (site2, self._occupancy[site1]))
                   #(sp1, sp2, site1, site2)
        else:
            # inefficient, maybe re-call method? infinite recursion problem
            return tuple(), tuple()

    def _get_swaps_from_table(self, swap_table=None):
        """Get a possible canonical flip, which is a swap between
        two sites based on a given swap table.

        Args:
            swap_table (dict): optional
                dictionary with keys identifying the possible swap types,
                including the species and sublattice of the two members of
                the swap type, i.e. (('Li+', 'Li+/Mn2+/Vacancy'), ('Li+',
                'Li+/Mn2+/Mn3+/Mn4+/Vacancy')). Instead of a sublattice, the
                keyword 'shared' can be specified for both members to indicate
                that the sites can be picked from any sublattice shared between
                the first and second species in the two flips. The values
                should be the probability at which the swap type is picked.
                The values must sum to 1.

        Returns: tuple

        """

        # TODO: helper function to aid in building swap table? care needs
        # to be taken by user that the given swap table satisfies detailed
        # balance. As a baseline, the following should definitely satisfy
        # detailed balance:
        # -Swaps between different species within a sublattice
        # -Swaps of different species over a set of shared sublattices
        # ('shared') or actually any subset of sublattices should work too
        # (although in this case, care needs to be taken that species do not
        # end up on the wrong sites) with a different species over all sublattices,
        # e.g. ('Li+', 'shared'), ('Mn2+', 'shared'). Given this, it may be
        # a good idea to allow lists of sublattices, although again this may
        # increase the burden on the user associated with creating this table

        # Create a swap_table if not given; the automatic swap table consists
        # only of swap types between different species in the same sublattice,
        # similar to what is given by the normal _get_flips method, all with
        # the same probability of being chosen
        if swap_table is None:
            if self.swap_table is None:
                self.swap_table = {}
                possible_sp = []
                for site_space in self.processor.unique_site_spaces:
                    possible_sp += site_space.keys()
                possible_sp = list(set(possible_sp))
                sp_sublatt_pairs = []
                for sp in possible_sp:
                    sp_str = str(sp)
                    for sublatt in self._active_sublatts:
                        if sp_str in self._active_sublatts[sublatt]['site_space'].keys() \
                                and len(self._site_table[sp_str][sublatt]) > 0:
                            sp_sublatt_pairs.append((sp_str, sublatt))
                allowed_swaps = [(pair1, pair2)
                                 for pair1, pair2 in itertools.combinations(sp_sublatt_pairs, 2)
                                 if pair1[0] != pair2[0] and pair1[1] == pair2[1]]
                for swap in allowed_swaps:
                    # for default option, give equal probability to each swap type
                    self.swap_table[swap] = 1.0/len(allowed_swaps)
                swap_table = self.swap_table
            else:
                swap_table = self.swap_table

        # Choose random swap type weighted by given probabilities in table
        chosen_flip = random.choices(list(swap_table.keys()),
                                     weights=list(swap_table.values()))[0]
        # Order shouldn't matter for independent distributions
        ((sp1, sublatt1), (sp2, sublatt2)) = chosen_flip

        if sublatt1 == 'shared' and sublatt2 == 'shared':
            sp1_sublatts = self._site_table[sp1].keys()
            sp2_sublatts = self._site_table[sp2].keys()
            shared_sublatts = list(set(sp1_sublatts) & set(sp2_sublatts))

            swap_options_1 = []
            swap_options_2 = []
            for sublatt in shared_sublatts:
                swap_options_1 += self._site_table[sp1][sublatt]
                swap_options_2 += self._site_table[sp2][sublatt]

            try:
                site1 = random.choice(swap_options_1)
                site2 = random.choice(swap_options_2)
            except:
                warnings.warn("At least one species does not exist on its given "
                              "sublattice in the list of possible flip types "
                              "(list of species on the sublattice is empty). "
                              "Continuing, returning an empty flip")
                #return tuple(), tuple()
                return tuple()

        else:
            site1 = random.choice(self._site_table[sp1][sublatt1])
            site2 = random.choice(self._site_table[sp2][sublatt2])

        # Use processor.allowed_species to ensure correct bit if sublattice changes
        return ((site1, self.processor.allowed_species[site1].index(sp2)),
                (site2, self.processor.allowed_species[site2].index(sp1)))
               #(sp1, sp2, site1, site2)

    def _get_Mn_swaps(self):
        """Get a possible canonical flip between Mn species, which
        can be either a swap or a disproportionation flip, resulting
        in a change of species.

        Returns: tuple

        """
        Mn_sp = ['Mn2+', 'Mn3+', 'Mn4+']
        site1_options = []
        for sp in Mn_sp:
            for sublatt in self._site_table[sp]:
                site1_options += self._site_table[sp][sublatt]
        if len(site1_options) < 2:
            raise ValueError("Only 1 Mn in the system. Cannot do Mn swaps.")
        site1 = random.choice(site1_options)

        # This implementation should still have p(s2) = 1/(N_Mn-1) for a given s2
        # and be faster than looking
        site2 = None
        while site2 is None:
            site2_proposal = random.choice(site1_options)
            if site2_proposal != site1:
                site2 = site2_proposal

        sp1 = self.processor.allowed_species[site1][self._occupancy[site1]]
        sp2 = self.processor.allowed_species[site2][self._occupancy[site2]]

        flip_type = random.choice(Mn_flip_table[(sp1, sp2)])

        if flip_type == 'None':
            # Unproductive swap, faster just to not return any flips
            return tuple()
        elif flip_type == 'swap':
            return ((site1, self.processor.allowed_species[site1].index(sp2)),
                    (site2, self.processor.allowed_species[site2].index(sp1)))
        elif flip_type == 'dispropA':
            return ((site1, self.processor.allowed_species[site1].index('Mn3+')),
                    (site2, self.processor.allowed_species[site2].index('Mn3+')))
        elif flip_type == 'dispropB':
            return ((site1, self.processor.allowed_species[site1].index('Mn2+')),
                    (site2, self.processor.allowed_species[site2].index('Mn4+')))
        elif flip_type == 'dispropC':
            return ((site1, self.processor.allowed_species[site1].index('Mn4+')),
                    (site2, self.processor.allowed_species[site2].index('Mn2+')))
        else:
            raise ValueError("No appropriate flip type in Mn flip table")
            return tuple()

    def _get_current_data(self):
        """Get ensemble specific data for current MC step."""
        data = super()._get_current_data()
        data['energy'] = self.current_energy
        return data

    def _update_site_table(self, swap):
        """Update site table based on a given swap."""
        if len(swap) == 2:
            site1 = swap[0][0]
            site2 = swap[1][0]
            sublatt1 = self._sites_to_sublattice[site1]
            sublatt2 = self._sites_to_sublattice[site2]
            sp1 = self.processor.allowed_species[site1][swap[0][1]]
            sp2 = self.processor.allowed_species[site2][swap[1][1]]

            # update self._site_table
            self._site_table[sp1][sublatt2][:] = \
                [x for x in self._site_table[sp1][sublatt2]
                 if x != site2]
            self._site_table[sp1][sublatt1].append(site1)
            self._site_table[sp2][sublatt1][:] = \
                [x for x in self._site_table[sp2][sublatt1]
                 if x != site1]
            self._site_table[sp2][sublatt2].append(site2)

    def _reset_site_table(self):
        """Set site table based on current occupancy."""
        self._site_table = {}
        possible_sp = []
        for site_space in self.processor.unique_site_spaces:
            possible_sp += site_space.keys()
        possible_sp = list(set(possible_sp))

        for sp in possible_sp:
            sp_str = str(sp)
            self._site_table[sp_str] = {}
            for sublatt in self._active_sublatts:
                if sp_str in self._active_sublatts[sublatt]['site_space'].keys():
                    self._site_table[sp_str][sublatt] = \
                        [i for i in self._active_sublatts[sublatt]['sites']
                         if self.processor.allowed_species[i][self._occupancy[i]] == sp_str]


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
