"""Implementations of MCMC Usher classes.

An usher is used to generate step proposals for MC Monte Carlo sampling.
For example a Flipper simply proposes a change of the identity of a species
at a site, for use in a SemiGrand ensemble. A Swapper will propose a swap
between species at two sites for use in Canonical ensemble simulations.

More complex steps can be defined simply by deriving from the MCUsher
"""

__author__ = "Luis Barroso-Luque, Fengyu Xie"

from abc import ABC, abstractmethod
import random
import numpy as np

from smol.utils import derived_class_factory
from ..comp_space import CompSpace

from ..utils.occu_utils import occu_to_species_stat, occu_to_species_list
from ..utils.comp_utils import get_n_links
from ..utils.math_utils import choose_section_from_partition, GCD_list


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

    def update_aux_state(self, step, *args, **kwargs):
        """Update any auxiliary state information based on an accepted step."""
        pass

    def set_aux_state(self, state, *args, **kwargs):
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


# The Combinedflip Usher
class Combinedflipper(MCMCUsher):
    """Implementation of combined flips on multiple sites.

    This is to further enable charge neutral unitary flips.
    """

    def __init__(self, sublattices, flip_combs, flip_weights=None):
        """Initialize Combinedflipper.

        Args:
            sublattices (list of Sublattice):
                list of Sublattices to propose steps for.
            flip_combs(List of Dict):
                A list of multi-site flips in dictionary format.
                For example,
                  [
                   {
                   'from':
                      {0(sublattice_id):
                         {0(specie_id in sublattice.species):
                              number_of_atoms_flipped,
                          1:
                              ...
                         },
                       1:
                         ...
                      },
                   'to':
                      ...
                   },
                   ...
                  ]
            flip_weights(1D Arraylike|Nonetype):
                Weights to adjust probability of each flip. If
                None given, will assign equal weights to each
                flip.
        """
        super().__init__(sublattices)
        self.flip_combs = flip_combs
        # A flip has two directions. All links initialized with 0.

        self.bits = [sl.species for sl in self.sublattices]
        self.sl_list = [list(sl.sites) for sl in self.sublattices]

        if flip_weights is None:
            self.flip_weights = np.ones(len(flip_combs), dtype=np.int64)
        else:
            if len(flip_weights) != len(flip_combs):
                raise ValueError("Flip reweighted, but not enough weights \
                                 supplied!")
            self.flip_weights = np.array(flip_weights)

    def propose_step(self, occupancy):
        """Propose a single random flip step.

        A step is given as a sequence of tuples, where each tuple is of the
        form (site index, species code to set)

        self.n_linkages are also updated here.
        Args:
            occupancy (ndarray):
                encoded occupancy string.

        Returns:
             Tuple(list(tuple),int): list of tuples each with (idex, code), and
                                   an integer indication flip direction in
                                   constrained coordinates. Different from
                                   other ushers!
        """
        occupancy = np.array(occupancy)

        species_stat = occu_to_species_stat(occupancy, self.bits, self.sl_list)
        n_links_current = get_n_links(species_stat, self.flip_combs)
        # Re-adjust by flip weights.
        flip_weights_tiled = np.array([self.flip_weights[i // 2]
                                      for i in range(2 *
                                                     len(self.flip_weights))])
        n_links_current = np.array(n_links_current) * flip_weights_tiled

        chosen_f_id = choose_section_from_partition(n_links_current)
        mult_flip = self.flip_combs[chosen_f_id // 2]
        direction = chosen_f_id % 2  # 0 forward, 1 backward.

        sp_list = occu_to_species_list(occupancy, self.bits, self.sl_list)

        chosen_sites_flip_from = [[] for sl in self.bits]
        chosen_sps_flip_to = [[] for sl in self.bits]

        if direction == 0:
            # Apply the forward direction
            for sl_id in mult_flip['from']:
                for sp_id in mult_flip['from'][sl_id]:
                    m_from = mult_flip['from'][sl_id][sp_id]
                    chosen_sites = random.sample(sp_list[sl_id][sp_id], m_from)
                    chosen_sites = sorted(chosen_sites)
                    # Remove duplicacy
                    chosen_sites_flip_from[sl_id].extend(chosen_sites)

            from_sites_for_sublats = [[i for i in range(len(sl))]
                                      for sl in chosen_sites_flip_from]

            for sl_id in mult_flip['to']:
                chosen_sps_flip_to[sl_id] = [None for st
                                             in chosen_sites_flip_from[sl_id]]
                for sp_id in mult_flip['to'][sl_id]:
                    m_to = mult_flip['to'][sl_id][sp_id]
                    from_sites = from_sites_for_sublats[sl_id]
                    chosen_sites = random.sample(from_sites, m_to)
                    chosen_sites = sorted(chosen_sites)
                    for st_id in chosen_sites:
                        chosen_sps_flip_to[sl_id][st_id] = sp_id
                        from_sites_for_sublats[sl_id].remove(st_id)

            for sl in from_sites_for_sublats:
                if len(sl) > 0:
                    raise ValueError("Flip not number conserved!")

        else:
            # Apply the backward direction
            for sl_id in mult_flip['to']:
                for sp_id in mult_flip['to'][sl_id]:
                    m_from = mult_flip['to'][sl_id][sp_id]
                    chosen_sites = random.sample(sp_list[sl_id][sp_id], m_from)
                    chosen_sites = sorted(chosen_sites)
                    # Remove duplicacy
                    chosen_sites_flip_from[sl_id].extend(chosen_sites)

            from_sites_for_sublats = [[i for i in range(len(sl))]
                                      for sl in chosen_sites_flip_from]

            for sl_id in mult_flip['from']:
                chosen_sps_flip_to[sl_id] = [None for st
                                             in chosen_sites_flip_from[sl_id]]
                for sp_id in mult_flip['from'][sl_id]:
                    m_to = mult_flip['from'][sl_id][sp_id]
                    from_sites = from_sites_for_sublats[sl_id]
                    chosen_sites = random.sample(from_sites, m_to)
                    chosen_sites = sorted(chosen_sites)
                    for st_id in chosen_sites:
                        chosen_sps_flip_to[sl_id][st_id] = sp_id
                        from_sites_for_sublats[sl_id].remove(st_id)

            for sl in from_sites_for_sublats:
                if len(sl) > 0:
                    raise ValueError("Flip not number conserved!")

        flip_list = []
        for sl_id, sl_sites in enumerate(chosen_sites_flip_from):
            for st_id, site in enumerate(sl_sites):
                flip_list.append((site, chosen_sps_flip_to[sl_id][st_id]))

        return flip_list


# The actual CN-SGMC flipper is a probabilistic combination of Swapper
# and CorrelatedFlipper
class Chargeneutralflipper(MCMCUsher):
    """Charge neutral flipper usher.

    This a combination of Swapper and CorrelatedFlipper.
    """

    def __init__(self, sublattices, flip_weights=None, n_links=None):
        """Initialize Chargeneutralflipper.

        Args:
            sublattices (list of Sublattice):
                list of Sublattices to propose steps for.

        Attributes:
            flip_weights(1D arrayLike|Nonetype):
                Weights to adjust probability of each flip. If None given,
                will assign equal weights to each flip.
            n_links(int):
                For each type of flip in the flip table, there is a number of
                possible flips of that type on the current occupancy. We store
                these numbers in the same order of flips in the flip_combs, and
                the numbers will be useful to keep a-priori reversibility
                in flips selection of the CN-SG ensemble.
        """
        super().__init__(sublattices)

        self.bits = [sl.species for sl in sublattices]
        self.sl_list = [list(sl.sites) for sl in sublattices]
        sl_sizes = [len(sl.sites) for sl in sublattices]

        # Here we use the GCD of sublattice sizes as supercell size. It
        # is not always the true supercell size, but always gives correct
        # compositional space.
        sc_size = GCD_list(sl_sizes)
        sl_sizes = [s // sc_size for s in sl_sizes]

        # self.comp_space and self.n_links contains complicated calculations.
        # Saving them elsewhere is highly recommended!
        compspace = CompSpace(self.bits, sl_sizes=sl_sizes)

        self._flip_combs = compspace.min_flips
        # This step may take quite some time.
        comp_vertices = compspace.int_vertices(sc_size=sc_size,
                                               form='compstat')

        if flip_weights is None:
            self.flip_weights = np.ones(len(self._flip_combs), dtype=np.int64)
        else:
            if len(flip_weights) != len(self._flip_combs):
                raise ValueError("Flip reweighted, but not enough weights \
                                 supplied!")
            self.flip_weights = np.array(flip_weights)

        if n_links is not None:
            # If n_links has been updated and computed before, utilize this
            self.n_links = n_links
        else:
            n_links_init = []
            for v in comp_vertices:
                n_links_current = get_n_links(v, self._flip_combs)
                flip_weights_tiled = np.array([self.flip_weights[i // 2]
                                               for i in range(2 *
                                               len(self.flip_weights))])

                n_links_init.append(sum(np.array(n_links_current) *
                                    flip_weights_tiled))

            self.n_links = max(n_links_init)

        self.swapper = Swapper(self.sublattices)
        self.comb_flipper = Combinedflipper(self.sublattices, self._flip_combs,
                                            self.flip_weights)

    def propose_step(self, occupancy):
        """Propose a single random swap step.

        A step is given as a sequence of tuples, where each tuple is of the
        form (site index, species code to set)

        self.n_links are also updated here after every step.

        Args:
            occupancy (ndarray):
                encoded occupancy string.

        Return:
            Tupe(list(tuple),int): list of tuples each with (idex, code), and
                                   an integer indication flip direction in
                                   constrained coordinates. Different from
                                   other ushers!
        """
        occupancy = np.array(occupancy)
        species_stat = occu_to_species_stat(occupancy, self.bits, self.sl_list)
        n_links_current = get_n_links(species_stat, self._flip_combs)
        # Re-adjust probabilities with flip weights.
        flip_weights_tiled = np.array([self.flip_weights[i // 2]
                                      for i in range(2 *
                                                     len(self.flip_weights))])
        n_links_current = np.array(n_links_current) * flip_weights_tiled

        if sum(n_links_current) > self.n_links:
            # Then update n_links value
            self.n_links = sum(n_links_current)

        # Assign a probabiltiy when we choose swaps instead of flips.
        p_swap = float(self.n_links - sum(n_links_current)) / self.n_links
        if random.random() < p_swap:
            # Choose a swap
            return self.swapper.propose_step(occupancy)
        else:
            # Choose a flip in table.
            return self.comb_flipper.propose_step(occupancy)


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
