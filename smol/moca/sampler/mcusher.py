"""Implementations of MCMC Usher classes.

An usher is used to generate step proposals for MC Monte Carlo sampling.
For example a Flipper simply proposes a change of the identity of a species
at a site, for use in a SemiGrand ensemble. A Swapper will propose a swap
between species at two sites for use in Canonical ensemble simulations.

More complex steps can be defined simply by deriving from the MCUsher
"""

__author__ = "Luis Barroso-Luque"

from abc import ABC, abstractmethod
import random
import numpy as np
import itertools
import warnings

from smol.utils import derived_class_factory
from pymatgen.core.periodic_table import Species


class MCMCUsher(ABC):
    """Abstract base class for MCMC usher classes."""

    def __init__(self, sublattices, sublattice_probabilities=None):
        """Initialize MCMCStep.

        Args:
            sublattices (list of Sublattice):
                list of Sublattices to propose steps for.
            sublattice_probabilities (list of float): optional
                list of probability to pick a site from a specific sublattice.
        """
        self.sublattices = sublattices
        if sublattice_probabilities is None:
            self._sublatt_probs = len(self.sublattices) * [1/len(self.sublattices), ]  # noqa
        elif len(sublattice_probabilities) != len(self.sublattices):
            raise AttributeError('Sublattice probabilites needs to be the '
                                 'same length as sublattices.')
        elif sum(sublattice_probabilities) != 1:
            raise ValueError('Sublattice probabilites must sum to one.')
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

    def update_aux_state(self, step):
        """Update any auxiliary state information based on an accepted step."""
        pass

    def set_aux_state(self, state, **kwargs):
        """Set the auxiliary state from a checkpoint values."""
        pass

    def get_random_sublattice(self):
        """Return a random sublattice based on given probabilities."""
        return random.choices(self.sublattices, weights=self._sublatt_probs)[0]


class Flipper(MCMCUsher):
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
        site = random.choice(sublattice.active_sites)
        choices = set(range(len(sublattice.site_space))) - {occupancy[site]}
        return [(site, random.choice(list(choices)))]


class Swapper(MCMCUsher):
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
        site1 = random.choice(sublattice.active_sites)
        species1 = occupancy[site1]
        sublattice_occu = occupancy[sublattice.active_sites]
        swap_options = sublattice.active_sites[sublattice_occu != species1]
        if swap_options.size > 0:  # check if swap_options are not empty
            site2 = random.choice(swap_options)
            swap = [(site1, occupancy[site2]), (site2, species1)]
        else:
            # inefficient, maybe re-call method? infinite recursion problem
            swap = []
        return swap


class Tableswapper(MCMCUsher):
    """
    Initialize Tableswapping algorithm from MCMCUsher base class.
    """

    def __init__(self, sublattices, allow_crossover=True,
                 swap_table=None, Mn_swap_probability=0.0):
        """
        Implementation of a swap step for two random sites within
        the same randomly chosen (shared or unshared) sublattice.
           Args:
            sublattices (list of Sublattice objects):
                list of Sublattices to propose steps for.
            allow_crossover: (bool, default = True). Allows swapping
            between sublattices which contain the same subset of species.
            For example, Li/Vac existing on tetrahedral and octahedral sites
            are allowed to swap between tetrahedral and octahedral sites.
            swap_table: (dict, default = None). Dictionary with keys being
            { ((Specie1, Sublattice1 SiteSpace),
               (Specie2, Sublattice2 SiteSpace)): probability (float),....}
            Mn_swap_probability: (float, default = 0.0).
            Gives the probability that a proposed swap is a Mn swap
            or a Mn disproportionation reaction
                """
        super(Tableswapper, self).__init__(sublattices)
        self.swap_table = swap_table
        self.allow_crossover = allow_crossover
        self.Mn_swap_probability = Mn_swap_probability
        self.Mn_flip_table = {('Mn2+', 'Mn2+'): ['None'],
                              ('Mn2+', 'Mn3+'): ['swap'],
                              ('Mn3+', 'Mn2+'): ['swap'],
                              ('Mn2+', 'Mn4+'): ['dispropA', 'swap'],
                              ('Mn4+', 'Mn2+'): ['dispropB', 'swap'],
                              ('Mn3+', 'Mn3+'): ['dispropC', 'dispropD'],
                              ('Mn3+', 'Mn4+'): ['swap'],
                              ('Mn4+', 'Mn3+'): ['swap'],
                              ('Mn4+', 'Mn4+'): ['None']}
        self.Mn2_specie = Species('Mn', 2)
        self.Mn3_specie = Species('Mn', 3)
        self.Mn4_specie = Species('Mn', 4)
        self.Mn_species = [self.Mn2_specie,
                           self.Mn3_specie,
                           self.Mn4_specie]
        self._sites_to_sublattice = None
        self.sublattice_probabilities_per_specie = None
        #self.swap_table = None
        self.current_flip_type = None
        self._site_table = None

    def _initialize_occupancies(self, occupancy):
        self._reset_site_table(occupancy)
        self._sites_to_sublattice = dict()
        for sublatt in self.sublattices:
            for site in sublatt.sites:
                self._sites_to_sublattice[site] = sublatt.site_space
        # now initialize sublattice specie probabilities
        self.sublattice_probabilities_per_specie = []
        for i, sublatt in enumerate(self.sublattices):
            for _ in list(sublatt.site_space):
                self.sublattice_probabilities_per_specie\
                    .append(self._sublatt_probs[i])

    def set_aux_state(self, state, **kwargs):
        self._initialize_occupancies(state)

    def _get_swaps_from_table(self, occupancy):
        """Args:

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

        TODO: helper function to aid in building swap table? care needs
         to be taken by user that the given swap table satisfies detailed
         balance. As a baseline, the following should definitely satisfy
         detailed balance:
         -Swaps between different species within a sublattice
         -Swaps of different species over a set of shared sublattices
         ('shared') or actually any subset of sublattices should work too
         (although in this case, care needs to be taken that species do not
         end up on the wrong sites) with a different species over all
         sublattices,e.g. ('Li+', 'shared'), ('Mn2+', 'shared').
         Given this, it may be a good idea to allow lists of sublattices,
         although again this may increase the burden on the user
         associated with creating this table

         Create a swap_table if not given; the automatic swap table consists
         only of swap types between different species in the same sublattice,
         similar to what is given by the normal _get_flips method, all with
         the same probability of being chosen
         """

        if self.swap_table is None:
            self._initialize_swap_table()
        # Choose random swap type weighted by given probabilities in table
        chosen_flip = random.choices(list(self.swap_table.keys()),
                                     weights=list(self.swap_table.values()))[0]
        # Order shouldn't matter for independent distributions
        ((sp1, sublatt1), (sp2, sublatt2)) = chosen_flip
        assert (sublatt1 == sublatt2)

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

                # Ensure correct bit if sublattice changes
                site2newIndex = list(self._sites_to_sublattice[site2]
                                     ).index(sp1)
                site1newIndex = list(self._sites_to_sublattice[site1]
                                     ).index(sp2)

                flip_type = 'crossover'

                return ((site1, site1newIndex),
                        (site2, site2newIndex)), flip_type

            except IndexError:
                warnings.warn("At least one species does not exist given "
                              "sublattice in list of possible flip types "
                              "(list of species on sublattice is empty). "
                              "Continuing, returning an empty flip")
                return tuple(), 'None'

        else:
            try:

                site1 = random.choice(self._site_table[sp1][sublatt1])
                site2 = random.choice(self._site_table[sp2][sublatt2])
            except IndexError:
                # we have no sites left in this sublattice.
                # one workaround is to update the site table...
                # but that might break d.b. Return None for now
                return tuple(), 'None'

        return ((site1, occupancy[site2]),
                (site2, occupancy[site1])), 'swap'

    def propose_step(self, occupancy):
        if self._site_table is None:
            self._initialize_occupancies(occupancy)
            # only need to set if we haven't made any flips yet
        if np.random.rand() <= self.Mn_swap_probability:
            flip, flip_type = self._get_Mn_swaps(occupancy)
        else:
            flip, flip_type = self._get_swaps_from_table(occupancy)
        self.current_flip_type = flip_type
        return flip

    def _reset_site_table(self, occupancy):
        """Set site table based on current occupancy."""
        self._site_table = {}
        possible_sp = []
        for sublattice in self.sublattices:
            site_space_list = list(sublattice.site_space)
            possible_sp += site_space_list
        possible_sp = [sp for sp in set(possible_sp)]

        for sp in possible_sp:
            self._site_table[sp] = {}
            for sublattice in self.sublattices:
                if sp in list(sublattice.site_space):  # noqa
                    self._site_table[sp][sublattice.site_space] = \
                        [i for i in sublattice.sites if
                         list(sublattice.site_space)[occupancy[i]] == sp]  # noqa

    def _initialize_swap_table(self):
        """

        Args:
            allow_crossover (bool): whether to allow swaps across
            sublattices for species with overlap between sublattices

        """
        self.swap_table = {}
        possible_sp = []
        for sublattice in self.sublattices:
            possible_sp += sublattice.site_space
        possible_sp = [sp for sp in set(possible_sp)]
        sp_sublatt_pairs = []
        for sp in possible_sp:
            # within each sublattice, allow swaps between diff species
            for sublatt in self.sublattices:
                ss_sp = sublatt.site_space.keys()  # noqa
                if sp in ss_sp and len(self._site_table[sp][sublatt.site_space]) > 0:  # noqa
                    sp_sublatt_pairs.append((sp, sublatt.site_space))
        allowed_swaps = [(pair1, pair2)
                         for pair1, pair2 in itertools.combinations(sp_sublatt_pairs, 2)  # noqa
                         if pair1[0] != pair2[0] and \
                         pair1[1] == pair2[1]]
        if self.allow_crossover:
            for sp1, sp2 in itertools.combinations(possible_sp, 2):
                # allow swaps within a set of shared sublattices
                sp1_sublatts = self._site_table[sp1].keys()
                sp2_sublatts = self._site_table[sp2].keys()
                shared_sublatts = list(set(sp1_sublatts) & set(sp2_sublatts))
                if len(shared_sublatts) > 1:
                    # if any list of sites would be empty, remove
                    # it from list of flip types to try
                    sp1_shared_num_sites = 0
                    sp2_shared_num_sites = 0
                    for sublatt in shared_sublatts:
                        sp1_shared_num_sites += \
                            len(self._site_table[sp1][sublatt])
                        sp2_shared_num_sites += \
                            len(self._site_table[sp2][sublatt])
                    if sp1_shared_num_sites == 0 or sp2_shared_num_sites == 0:
                        continue
                    allowed_swaps.append(((sp1, 'shared'), (sp2, 'shared')))
                    # remove extra swaps between species with shared
                    # sublattices that are only between single sublattices
                    for sublatt in shared_sublatts:
                        to_remove = [((sp1, sublatt), (sp2, sublatt)),
                                     ((sp2, sublatt), (sp1, sublatt))]
                        allowed_swaps = [x for x in allowed_swaps
                                         if x not in to_remove]
        for swap in allowed_swaps:
            self.swap_table[swap] = 1.0 / len(allowed_swaps)

    def update_aux_state(self, flip):
        """Update site_table."""
        flip_type = self.current_flip_type
        self._update_site_table(flip, flip_type)

    def _update_site_table(self, flip, flip_type):
        """Update site table based on a given swap."""
        if len(flip) == 2:
            site1 = flip[0][0]
            site2 = flip[1][0]
            sublatt1 = self._sites_to_sublattice[site1]
            sublatt2 = self._sites_to_sublattice[site2]

            sp2 = list(sublatt1)[flip[0][1]]
            sp1 = list(sublatt2)[flip[1][1]]

            # update self._site_table
            # identify correct old species if there's a disprop swap

            if flip_type == 'dispropA':
                # note these are opposite from the Mn table because the method
                # below assumes swapping.
                old_sp1 = self.Mn2_specie
                old_sp2 = self.Mn4_specie
            elif flip_type == 'dispropB':
                old_sp1 = self.Mn4_specie
                old_sp2 = self.Mn2_specie
            elif flip_type == 'dispropC' or flip_type == 'dispropD':
                old_sp1 = self.Mn3_specie
                old_sp2 = self.Mn3_specie
            else:
                old_sp1 = sp1
                old_sp2 = sp2
                #sublatt1, sublatt2 = sublatt2, sublatt1
                # crucial step! was not here in v0.0
            #else:
            #    old_sp1 = list(sublatt1)[flip[1][1]]
            #    old_sp2 = list(sublatt2)[flip[0][1]]

            # remove old species from table
            self._site_table[old_sp1][sublatt1][:] = \
                [x for x in self._site_table[old_sp1][sublatt1]
                 if x != site1]
            self._site_table[old_sp2][sublatt2][:] = \
                [x for x in self._site_table[old_sp2][sublatt2]
                 if x != site2]

            # add new species to table
            if flip_type == 'crossover':
                self._site_table[sp1][sublatt2].append(site2)
                self._site_table[sp2][sublatt1].append(site1)
            else:
                self._site_table[sp1][sublatt1].append(site2)
                self._site_table[sp2][sublatt2].append(site1)

    def _get_Mn_swaps(self, occupancy):
        """
        Get a possible swap between Mn species.

        Make all Mn on the same sublattice to satisfy detailed balance.

        The Mn swap can be either a swap or a disproportionation flip,
        resulting in a change of species.

        Returns: tuple of (flip, flip_type)
        flip: ((site1, sp2 index for site1 sublattice),
               (site2, sp1 index for site2 sublattice))
        flip_type: (str) which can be 'swap', 'crossover',
                    None (inefficient but satisfies d.b.),
                    dispropRXN.
        """

        site1_options = []
        for sp in self.Mn_species:
            for sublatt in self._site_table[sp]:
                site1_options += self._site_table[sp][sublatt]
        if len(site1_options) < 2:
            raise ValueError("Only 1 Mn in the system. \
                            Cannot do Mn swaps.")

        site1 = random.choice(site1_options)

        # This implementation should still have p(s2) = 1/(N_Mn-1) for a
        # given s2 and be faster than looking
        site2 = None
        while site2 is None:
            site2_proposal = random.choice(site1_options)
            if site2_proposal != site1:
                site2 = site2_proposal
        sp2 = list(self._sites_to_sublattice[site2])[occupancy[site2]]
        sp1 = list(self._sites_to_sublattice[site1])[occupancy[site1]]

        flip_type = random.choice(self.Mn_flip_table[(str(sp1),
                                                      str(sp2))])

        if flip_type == 'None':
            # Unproductive swap, faster just to not return any flips
            return tuple(), 'None'
        elif flip_type == 'swap':
            try:
                return ((site1,
                         list(self._sites_to_sublattice[site1])).index(sp2),
                        (site2,
                         list(self._sites_to_sublattice[site2])).index(sp1)),\
                        flip_type
            except ValueError:  # Mn3+/4+ could go tetrahedral
                return tuple(), flip_type
        elif flip_type == 'dispropA' or flip_type == 'dispropB':
            try:
                return ((site1,
                         list(self._sites_to_sublattice[site1])
                         .index(self.Mn3_specie)),
                        (site2,
                         list(self._sites_to_sublattice[site2])
                         .index(self.Mn3_specie))), \
                       flip_type
            except ValueError:
                return tuple(), 'None'
        elif flip_type == 'dispropC':

            return ((site1,
                     list(self._sites_to_sublattice[site1])
                     .index(self.Mn2_specie)),
                    (site2,
                     list(self._sites_to_sublattice[site2])
                     .index(self.Mn4_specie))), \
                   flip_type
        elif flip_type == 'dispropD':
            return ((site1,
                     list(self._sites_to_sublattice[site1]).
                     index(self.Mn4_specie)),
                    (site2,
                     list(self._sites_to_sublattice[site2])
                     .index(self.Mn2_specie))), \
                   flip_type
        else:
            raise ValueError("No appropriate flip type in Mn flip table")

    def _normalize_swap_table(self):
        """Normalize swap table so values sum to 1."""
        sum_probs = np.sum([self.swap_table[flip] for flip in self.swap_table])
        for flip in self.swap_table:
            self.swap_table[flip] = self.swap_table[flip]/sum_probs

class Grandcanonicaltableswapper(MCMCUsher):
    """
    Initialize Tableswapping algorithm from MCMCUsher base class.
    """

    def __init__(self, sublattices, allow_crossover=True,
                 GC_swap_probability=0.0, swap_table = None,
                 gc_swap_table = None):
        """
        Implementation of a swap step for two random sites within
        the same randomly chosen (shared or unshared) sublattice.
           Args:
            sublattices (list of Sublattice objects):
                list of Sublattices to propose steps for.
            allow_crossover: (bool, default = True). Allows swapping
            between sublattices which contain the same subset of species.
            For example, Li/Vac existing on tetrahedral and octahedral sites
            are allowed to swap between tetrahedral and octahedral sites.
            swap_table: (dict, default = None). Dictionary with keys being
            { ((Specie1, Sublattice1 SiteSpace),
               (Specie2, Sublattice2 SiteSpace)): probability (float),....}
            Mn_swap_probability: (float, default = 0.0).
            Gives the probability that a proposed swap is a Mn swap
            or a Mn disproportionation reaction
                """
        super(Grandcanonicaltableswapper, self).__init__(sublattices)
        self.swap_table = swap_table
        self.allow_crossover = allow_crossover
        self.gc_probability = GC_swap_probability  # rename default variables
        self.gc_swap_table = gc_swap_table
        self.Mn_flip_table = {('Mn2+', 'Mn2+'): ['None'],
                              ('Mn2+', 'Mn3+'): ['swap'],
                              ('Mn3+', 'Mn2+'): ['swap'],
                              ('Mn2+', 'Mn4+'): ['dispropA', 'swap'],
                              ('Mn4+', 'Mn2+'): ['dispropB', 'swap'],
                              ('Mn3+', 'Mn3+'): ['dispropC', 'dispropD'],
                              ('Mn3+', 'Mn4+'): ['swap'],
                              ('Mn4+', 'Mn3+'): ['swap'],
                              ('Mn4+', 'Mn4+'): ['None']}
        self.Mn2_specie = Species('Mn', 2)
        self.Mn3_specie = Species('Mn', 3)
        self.Mn4_specie = Species('Mn', 4)
        self.Mn_species = [self.Mn2_specie,
                           self.Mn3_specie,
                           self.Mn4_specie]
        self._sites_to_sublattice = None
        self.sublattice_probabilities_per_specie = None
        self.current_flip_type = None
        self._site_table = None
        self.oldsp1 = None
        self.oldsp2 = None
        self._initialize_flips_to_sublattice()

    def _initialize_occupancies(self, occupancy):
        self._reset_site_table(occupancy)
        self._sites_to_sublattice = dict()
        for sublatt in self.sublattices:
            for site in sublatt.sites:
                self._sites_to_sublattice[site] = sublatt.site_space
        # now initialize sublattice specie probabilities
        self.sublattice_probabilities_per_specie = []
        for i, sublatt in enumerate(self.sublattices):
            for _ in list(sublatt.site_space):
                self.sublattice_probabilities_per_specie\
                    .append(self._sublatt_probs[i])

    def set_aux_state(self, state, **kwargs):
        self._initialize_occupancies(state)

    def _get_swaps_from_table(self, occupancy):
        """Args:

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

        TODO: helper function to aid in building swap table? care needs
         to be taken by user that the given swap table satisfies detailed
         balance. As a baseline, the following should definitely satisfy
         detailed balance:
         -Swaps between different species within a sublattice
         -Swaps of different species over a set of shared sublattices
         ('shared') or actually any subset of sublattices should work too
         (although in this case, care needs to be taken that species do not
         end up on the wrong sites) with a different species over all
         sublattices,e.g. ('Li+', 'shared'), ('Mn2+', 'shared').
         Given this, it may be a good idea to allow lists of sublattices,
         although again this may increase the burden on the user
         associated with creating this table

         Create a swap_table if not given; the automatic swap table consists
         only of swap types between different species in the same sublattice,
         similar to what is given by the normal _get_flips method, all with
         the same probability of being chosen
         """

        if self.swap_table is None:
            self._initialize_swap_table()
        # Choose random swap type weighted by given probabilities in table
        chosen_flip = random.choices(list(self.swap_table.keys()),
                                     weights=list(self.swap_table.values()))[0]
        # Order shouldn't matter for independent distributions
        ((sp1, sublatt1), (sp2, sublatt2)) = chosen_flip
        assert (sublatt1 == sublatt2)

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

                # Ensure correct bit if sublattice changes
                site2newIndex = list(self._sites_to_sublattice[site2]
                                     ).index(sp1)
                site1newIndex = list(self._sites_to_sublattice[site1]
                                     ).index(sp2)

                flip_type = 'crossover'

                return ((site1, site1newIndex),
                        (site2, site2newIndex)), flip_type

            except IndexError:
                warnings.warn("At least one species does not exist given "
                              "sublattice in list of possible flip types "
                              "(list of species on sublattice is empty). "
                              "Continuing, returning an empty flip")
                return tuple(), 'None'

        else:
            try:

                site1 = random.choice(self._site_table[sp1][sublatt1])
                site2 = random.choice(self._site_table[sp2][sublatt2])
            except IndexError:
                # we have no sites left in this sublattice.
                # one workaround is to update the site table...
                # but that might break d.b. Return None for now
                return tuple(), 'None'

        return ((site1, occupancy[site2]),
                (site2, occupancy[site1])), 'swap'

    def propose_step(self, occupancy):
        if self._site_table is None:
            self._initialize_occupancies(occupancy)
            # only need to set if we haven't made any flips yet
        if np.random.rand() <= self.gc_probability:  # Grand canonical loop
            flip, flip_type = self._do_species_change_flip(occupancy)
        else:  # regular canonical loop
            flip, flip_type = self._get_swaps_from_table(occupancy)
        self.current_flip_type = flip_type
        return flip

    def _reset_site_table(self, occupancy):
        """Set site table based on current occupancy."""
        self._site_table = {}
        possible_sp = []
        for sublattice in self.sublattices:
            site_space_list = list(sublattice.site_space)
            possible_sp += site_space_list
        possible_sp = [sp for sp in set(possible_sp)]

        for sp in possible_sp:
            self._site_table[sp] = {}
            for sublattice in self.sublattices:
                if sp in list(sublattice.site_space):  # noqa
                    self._site_table[sp][sublattice.site_space] = \
                        [i for i in sublattice.sites if
                         list(sublattice.site_space)[occupancy[i]] == sp]  # noqa

    def _initialize_swap_table(self):
        """

        Args:
            allow_crossover (bool): whether to allow swaps across
            sublattices for species with overlap between sublattices

        """
        self.swap_table = {}
        possible_sp = []
        for sublattice in self.sublattices:
            possible_sp += sublattice.site_space
        possible_sp = [sp for sp in set(possible_sp)]
        sp_sublatt_pairs = []
        for sp in possible_sp:
            # within each sublattice, allow swaps between diff species
            for sublatt in self.sublattices:
                ss_sp = sublatt.site_space.keys()  # noqa
                if sp in ss_sp and len(self._site_table[sp][sublatt.site_space]) > 0:  # noqa
                    sp_sublatt_pairs.append((sp, sublatt.site_space))
        allowed_swaps = [(pair1, pair2)
                         for pair1, pair2 in itertools.combinations(sp_sublatt_pairs, 2)  # noqa
                         if pair1[0] != pair2[0] and \
                         pair1[1] == pair2[1]]
        if self.allow_crossover:
            for sp1, sp2 in itertools.combinations(possible_sp, 2):
                # allow swaps within a set of shared sublattices
                sp1_sublatts = self._site_table[sp1].keys()
                sp2_sublatts = self._site_table[sp2].keys()
                shared_sublatts = list(set(sp1_sublatts) & set(sp2_sublatts))
                if len(shared_sublatts) > 1:
                    # if any list of sites would be empty, remove
                    # it from list of flip types to try
                    sp1_shared_num_sites = 0
                    sp2_shared_num_sites = 0
                    for sublatt in shared_sublatts:
                        sp1_shared_num_sites += \
                            len(self._site_table[sp1][sublatt])
                        sp2_shared_num_sites += \
                            len(self._site_table[sp2][sublatt])
                    if sp1_shared_num_sites == 0 or sp2_shared_num_sites == 0:
                        continue
                    allowed_swaps.append(((sp1, 'shared'), (sp2, 'shared')))
                    # remove extra swaps between species with shared
                    # sublattices that are only between single sublattices
                    for sublatt in shared_sublatts:
                        to_remove = [((sp1, sublatt), (sp2, sublatt)),
                                     ((sp2, sublatt), (sp1, sublatt))]
                        allowed_swaps = [x for x in allowed_swaps
                                         if x not in to_remove]
        for swap in allowed_swaps:
            self.swap_table[swap] = 1.0 / len(allowed_swaps)

    def update_aux_state(self, flip):
        """Update site_table."""
        flip_type = self.current_flip_type
        self._update_site_table(flip, flip_type)

    def _update_site_table(self, flip, flip_type):
        """Update site table based on a given swap."""
        if len(flip) == 2:
            site1 = flip[0][0]
            site2 = flip[1][0]
            sublatt1 = self._sites_to_sublattice[site1]
            sublatt2 = self._sites_to_sublattice[site2]

            sp2 = list(sublatt1)[flip[0][1]]
            sp1 = list(sublatt2)[flip[1][1]]

            # update self._site_table
            # identify correct old species if there's a disprop swap

            if flip_type == 'dispropA':
                # note these are opposite from the Mn table because the method
                # below assumes swapping.
                old_sp1 = self.Mn2_specie
                old_sp2 = self.Mn4_specie
            elif flip_type == 'dispropB':
                old_sp1 = self.Mn4_specie
                old_sp2 = self.Mn2_specie
            elif flip_type == 'dispropC' or flip_type == 'dispropD':
                old_sp1 = self.Mn3_specie
                old_sp2 = self.Mn3_specie
            elif flip_type == 'semi-grand':
                old_sp1 = self.oldsp1
                old_sp2 = self.oldsp2
            else:
                old_sp1 = sp1
                old_sp2 = sp2
                #sublatt1, sublatt2 = sublatt2, sublatt1
                # crucial step! was not here in v0.0
            #else:
            #    old_sp1 = list(sublatt1)[flip[1][1]]
            #    old_sp2 = list(sublatt2)[flip[0][1]]

            # remove old species from table
            self._site_table[old_sp1][sublatt1][:] = \
                [x for x in self._site_table[old_sp1][sublatt1]
                 if x != site1]
            self._site_table[old_sp2][sublatt2][:] = \
                [x for x in self._site_table[old_sp2][sublatt2]
                 if x != site2]

            # add new species to table
            if flip_type == 'crossover':
                self._site_table[sp1][sublatt2].append(site2)
                self._site_table[sp2][sublatt1].append(site1)
            else:
                self._site_table[sp1][sublatt1].append(site2)
                self._site_table[sp2][sublatt2].append(site1)

    def _get_Mn_swaps(self, occupancy):
        """
        Get a possible swap between Mn species.

        Make all Mn on the same sublattice to satisfy detailed balance.

        The Mn swap can be either a swap or a disproportionation flip,
        resulting in a change of species.

        Returns: tuple of (flip, flip_type)
        flip: ((site1, sp2 index for site1 sublattice),
               (site2, sp1 index for site2 sublattice))
        flip_type: (str) which can be 'swap', 'crossover',
                    None (inefficient but satisfies d.b.),
                    dispropRXN.
        """

        site1_options = []
        for sp in self.Mn_species:
            for sublatt in self._site_table[sp]:
                site1_options += self._site_table[sp][sublatt]
        if len(site1_options) < 2:
            raise ValueError("Only 1 Mn in the system. \
                            Cannot do Mn swaps.")

        site1 = random.choice(site1_options)

        # This implementation should still have p(s2) = 1/(N_Mn-1) for a
        # given s2 and be faster than looking
        site2 = None
        while site2 is None:
            site2_proposal = random.choice(site1_options)
            if site2_proposal != site1:
                site2 = site2_proposal
        sp2 = list(self._sites_to_sublattice[site2])[occupancy[site2]]
        sp1 = list(self._sites_to_sublattice[site1])[occupancy[site1]]

        flip_type = random.choice(self.Mn_flip_table[(str(sp1),
                                                      str(sp2))])

        if flip_type == 'None':
            # Unproductive swap, faster just to not return any flips
            return tuple(), 'None'
        elif flip_type == 'swap':
            try:
                return ((site1,
                         list(self._sites_to_sublattice[site1])).index(sp2),
                        (site2,
                         list(self._sites_to_sublattice[site2])).index(sp1)),\
                        flip_type
            except ValueError:  # Mn3+/4+ could go tetrahedral
                return tuple(), flip_type
        elif flip_type == 'dispropA' or flip_type == 'dispropB':
            try:
                return ((site1,
                         list(self._sites_to_sublattice[site1])
                         .index(self.Mn3_specie)),
                        (site2,
                         list(self._sites_to_sublattice[site2])
                         .index(self.Mn3_specie))), \
                       flip_type
            except ValueError:
                return tuple(), 'None'
        elif flip_type == 'dispropC':

            return ((site1,
                     list(self._sites_to_sublattice[site1])
                     .index(self.Mn2_specie)),
                    (site2,
                     list(self._sites_to_sublattice[site2])
                     .index(self.Mn4_specie))), \
                   flip_type
        elif flip_type == 'dispropD':
            return ((site1,
                     list(self._sites_to_sublattice[site1]).
                     index(self.Mn4_specie)),
                    (site2,
                     list(self._sites_to_sublattice[site2])
                     .index(self.Mn2_specie))), \
                   flip_type
        else:
            raise ValueError("No appropriate flip type in Mn flip table")

    def _normalize_swap_table(self):
        """Normalize swap table so values sum to 1."""
        sum_probs = np.sum([self.swap_table[flip] for flip in self.swap_table])
        for flip in self.swap_table:
            self.swap_table[flip] = self.swap_table[flip]/sum_probs

    def _get_sublattice_for_flip(self, chosen_gc_flip):
        """
        Wrapper function to return the sites from which
        to pick a flip, given a site-space of interest.
        To maximize efficiency, we narrow in on just the sublattice which
        currently has the flips of interest.
        For example, if we are interested in a (Li, Mn2+) -> (Vac, Mn3+)
        flip, there is no reason to add a site with Mn4+ to the list of
        possible flips.
        :param sublattice_to_pick: site space
        :return: list of sites
        """
        site_space = self.flip_to_sublattice[chosen_gc_flip]
        if chosen_gc_flip == 'Mn_disproportionation':
            sp_list = self.Mn_species
        else:
            sp_list = [sp for flip in chosen_gc_flip for sp in flip]
        site_options = []
        for sp in sp_list:
            site_options += self._site_table[sp][site_space]
        if len(site_options) < 2:
            raise ValueError("Cannot do swaps because there is "
                             "only one species in specie list", str(sp_list),
                             "in the structure.")
        return site_options

    def _initialize_flips_to_sublattice(self):
        """
        Initialize the common site space with which flips can occur.
        For example (Li, Mn2+) -> (Vac, Mn3+) can only happen on the
        octahedral site space if Mn3+ cannot be tetrahedral even if
        Li, Mn2+, Vacancy can be tetrahedral.
        :return: None
        """
        self.flip_to_sublattice = dict()
        for flip in self.gc_swap_table:
            # hard coded the Mn disproportionation case.
            # TODO: make the Mn_disproportionation feature more general
            if flip == 'Mn_disproportionation':
                flattened = [self.Mn2_specie, self.Mn3_specie, self.Mn4_specie]
            else:
                flattened = [item for sublist in flip for item in sublist]
            for sublatt in self.sublattices:
                if np.alltrue([i in sublatt.site_space for i in flattened]):
                    self.flip_to_sublattice[flip] = sublatt.site_space
                    break
        # assert that all gc flips have a sublattice
        assert len(self.flip_to_sublattice) == len(self.gc_swap_table)

    def _do_species_change_flip(self, occupancy):
        """
        Pick a random GC flip. Similar to the canonical-table-swap method
        where a flip is picked based on the keys of the swap_table
        :param occupancy:
        :return: (site1, newsp1), (site2, newsp2) in accordance with a
        semi-grand canonical, charge-balanced flip
        """
        # Choose random GC swap type weighted by given probabilities in table
        chosen_gc_flip = random.choices(list(self.gc_swap_table.keys()),
                                     weights=list(self.gc_swap_table.values()))[0]


        site1_options = self._get_sublattice_for_flip(chosen_gc_flip)
        site1 = random.choice(site1_options)
        # This implementation should still have p(s2) = 1/(N_Mn-1) for a
        # given s2 and be faster than looking
        site2 = None
        while site2 is None:
            site2_proposal = random.choice(site1_options)
            if site2_proposal != site1:
                site2 = site2_proposal
        sp2 = list(self._sites_to_sublattice[site2])[occupancy[site2]]
        sp1 = list(self._sites_to_sublattice[site1])[occupancy[site1]]

        if chosen_gc_flip == 'Mn_disproportionation':
            return self._do_Mn_flip(sp1, sp2, site1, site2)
        # charge-balanced sGC
        return self._do_sg_flip(chosen_gc_flip, sp1, sp2, site1, site2)

    def _do_sg_flip(self, chosen_gc_flip, sp1, sp2, site1, site2):
        """
        Does the semi-grand-canonical specie swap given a sublattice
        and the types of allowable swaps have already been specified.
        Only allow Li/Vac specie and Mn2/3/4 species swaps. This means that
        a picked deliathiation move will not have Mn3+ swapping for a Li+
        and a Mn2+ swapping for a Li+.
        :param chosen_gc_flip: the semi-grand flip that is being done
        :param sp1: old specie 1 on site 1 to be swapped out
        :param sp2: old specie 2 on site 2 to be swapped out
        :param site1: site1 to have a new specie 1
        :param site2: site2 to have a new specie 2
        :return: ((site1, newsp1), (site2, newsp2)), 'semi-grand'
        """
        chosen = [0, 1]
        for i, flip in enumerate(chosen_gc_flip):
            if sp1 in flip and sp2 in flip and sp1 != sp2:
                chosen.pop(i)
                sp1index = flip.index(sp1)  # save the index of sp1
                sp2index = flip.index(sp2)  # save the index of sp2
                self.oldsp1 = chosen_gc_flip[i][sp1index]
                self.oldsp2 = chosen_gc_flip[i][sp2index]
        if len(chosen) == 2:
            # selected species are not ones we can flip.
            # Best way now to ensure detailed balance is to return None
            return tuple(), None
        assert len(chosen) == 1
        newsp1 = chosen_gc_flip[chosen[0]][sp1index]
        newsp2 = chosen_gc_flip[chosen[0]][sp2index]
        #print ('old-sp1', self.oldsp1, 'new:', newsp1, site1)
        #print('old-sp2', self.oldsp2, 'new:', newsp2, site2)
        return ((site1, list(self._sites_to_sublattice[site1]).index(newsp1)),
                (site2, list(self._sites_to_sublattice[site2]).index(newsp2))),\
                'semi-grand'

    def _do_Mn_flip(self, sp1, sp2, site1, site2):
        flip_type = random.choice(self.Mn_flip_table[(str(sp1),
                                                      str(sp2))])

        if flip_type == 'None':
            # Unproductive swap, faster just to not return any flips
            return tuple(), 'None'
        elif flip_type == 'swap':
            try:
                return ((site1,
                         list(self._sites_to_sublattice[site1])).index(sp2),
                        (site2,
                         list(self._sites_to_sublattice[site2])).index(sp1)), \
                       flip_type
            except ValueError:  # Mn3+/4+ could go tetrahedral
                return tuple(), flip_type
        elif flip_type == 'dispropA' or flip_type == 'dispropB':
            try:
                return ((site1,
                         list(self._sites_to_sublattice[site1])
                         .index(self.Mn3_specie)),
                        (site2,
                         list(self._sites_to_sublattice[site2])
                         .index(self.Mn3_specie))), \
                       flip_type
            except ValueError:
                return tuple(), 'None'
        elif flip_type == 'dispropC':

            return ((site1,
                     list(self._sites_to_sublattice[site1])
                     .index(self.Mn2_specie)),
                    (site2,
                     list(self._sites_to_sublattice[site2])
                     .index(self.Mn4_specie))), \
                   flip_type
        elif flip_type == 'dispropD':
            return ((site1,
                     list(self._sites_to_sublattice[site1]).
                     index(self.Mn4_specie)),
                    (site2,
                     list(self._sites_to_sublattice[site2])
                     .index(self.Mn2_specie))), \
                   flip_type
        else:
            raise ValueError("No appropriate flip type in Mn flip table")

def mcusher_factory(usher_type, sublattices, *args, **kwargs):
    """Get a MCMC Usher from string name.

    Args:
        usher_type (str):
            string specifying step to instantiate.
        sublattices (list of Sublattice):
                list of Sublattices to propose steps for.
        *args:
            positional arguments passed to class constructor
        **kwargs:
            Keyword arguments passed to class constructor

    Returns:
        MCMCUsher: instance of derived class.
    """
    return derived_class_factory(usher_type.capitalize(), MCMCUsher,
                                 sublattices, *args, **kwargs)
