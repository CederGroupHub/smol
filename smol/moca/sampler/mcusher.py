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


# The TableFlip Usher
class Tableflipper(MCMCUsher):
    """Implementation of table flips.

    Use user-specified or auto-calculated flip tables. Does not
    check the correctness of the flip tables.

    No longer computing and assigning a-priori proabilities
    before flip selection, and selects sites instead.

    Chargeneutralflipper is implemented based on this class.

    Direct use of this class is not recommended, because the
    dict form of the flip table is not very easy to read and
    write.
    """

    def __init__(self, sublattices, flip_table, flip_weights=None):
        """Initialize Tableflipper.

        Args:
            sublattices (list of Sublattice):
                list of Sublattices to propose steps for.
            flip_table(List of Dict):
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
        self.flip_table = flip_table
        # A flip has two directions. All links initialized with 0.

        self.bits = [sl.species for sl in self.sublattices]
        self.sl_list = [list(sl.sites) for sl in self.sublattices]

        if flip_weights is None:
            self.flip_weights = np.ones(len(flip_table), dtype=np.int64)
        else:
            if len(flip_weights) != len(flip_table):
                raise ValueError("Flip reweighted, but not enough weights \
                                 supplied!")
            self.flip_weights = np.array(flip_weights)

    def propose_step(self, occupancy):
        """Propose a single random flip step.

        A step is given as a sequence of tuples, where each tuple is of the
        form (site index, species code to set)

        1, Pick a flip in table;
        2, Pick sites;
        3, See if sites have species in either side of the picked flip.
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

        picked_flip = self.flip_table[choose_section_from_partition(
                                      self.flip_weights)]

        # Pick sites
        species_list = occu_to_species_list(occupancy, self.bits,
                                            self.sl_list)
        flip_list = []
        direction = None

        for sl_id, sl_sites in enumerate(species_list):
            if sl_id not in picked_flip['from']:
                continue

            union_sp_ids = (list(picked_flip['from'][sl_id].keys()) +
                            list(picked_flip['to'][sl_id].keys()))
            n_picks = sum(list(picked_flip['from'][sl_id].values()))
            pickable_sites = []
            for sp_id in union_sp_ids:
                pickable_sites.extend(sl_sites[sp_id])

            picked_sites_sl = random.sample(pickable_sites, n_picks)
            picked_table = {}
            for s_id in picked_sites_sl:
                sp_id = None
                for j, sp_sites in enumerate(sl_sites):
                    if s_id in sp_sites:
                        sp_id = j
                        break
                if sp_id not in picked_table:
                    picked_table[sp_id] = 1
                else:
                    picked_table[sp_id] += 1

            # Confirm direction.
            if direction is None:
                if (picked_table != picked_flip['from'][sl_id] and
                        picked_table != picked_flip['to'][sl_id]):
                    # Site selection not successful.
                    flip_list = []
                    break
                elif picked_table == picked_flip['from'][sl_id]:
                    flip_to_table = picked_flip['to'][sl_id]
                    direction = 1
                else:
                    flip_to_table = picked_flip['from'][sl_id]
                    direction = -1

            elif direction > 0:
                if picked_table == picked_flip['from'][sl_id]:
                    flip_to_table = picked_flip['to'][sl_id]
                else:
                    flip_list = []
                    break

            else:
                if picked_table == picked_flip['to'][sl_id]:
                    flip_to_table = picked_flip['from'][sl_id]
                else:
                    flip_list = []
                    break

            n_flipped = 0
            flip_failed = False
            for sp_id, n_flip in flip_to_table.items():
                flip_sites = picked_sites_sl[n_flipped: n_flipped + n_flip]
                # This check is required to keep equal-a-priori.
                if flip_sites != sorted(flip_sites):
                    flip_list = []
                    flip_failed = True
                    break
                flip_list.extend([(s_id, sp_id) for s_id in flip_sites])
                n_flipped += n_flip

            if flip_failed:
                break

            if n_flipped != n_picks:
                raise ValueError("Flip is not number conserved!")

        # If no valid flip is picked, try a canonical swap.
        if len(flip_list) == 0:
            active_sublattices = [sl for sl in self.sublattices
                                  if not np.all(occupancy[sl.sites] ==
                                                occupancy[sl.sites][0])]
            if len(active_sublattices) != 0:
                _swapper = Swapper(active_sublattices)
                flip_list = _swapper.propose_step(occupancy)

        return flip_list


class Chargeneutralflipper(Tableflipper):
    """Charge neutral flipper usher.

    Implemented based on Tableflipper. Uses minial, sublattice
    discriminative flip table.
    """

    def __init__(self, sublattices, flip_weights=None):
        """Initialize Chargeneutralflipper.

        Args:
            sublattices (list of Sublattice):
                list of Sublattices to propose steps for.

        Attributes:
            flip_weights(1D arrayLike|Nonetype):
                Weights to adjust probability of each flip. If None given,
                will assign equal weights to each flip.
                Length must equal to the dimensionality of CompSpace.
        """
        bits = [sl.species for sl in sublattices]
        sl_list = [list(sl.sites) for sl in sublattices]
        sl_sizes = [len(sl.sites) for sl in sublattices]

        sc_size = GCD_list(sl_sizes)
        sl_sizes = [s // sc_size for s in sl_sizes]

        flip_table = CompSpace(bits, sl_sizes).min_flip_table

        super().__init__(sublattices, flip_table, flip_weights=flip_weights)


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
    if usher_type.capitalize() == 'Chargeneutralflipper':
        return derived_class_factory(usher_type.capitalize(),
                                     Tableflipper, sublattices,
                                     *args, **kwargs)

    return derived_class_factory(usher_type.capitalize(), MCMCUsher,
                                 sublattices, *args, **kwargs)
