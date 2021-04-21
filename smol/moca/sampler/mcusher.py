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
from .bias import mcbias_factory
from ..comp_space import CompSpace

from ..utils.occu_utils import occu_to_species_list
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

    Direct use of this class is not recommended, because the
    dict form of the flip table is not very easy to read and
    write.
    """

    def __init__(self, all_sublattices, flip_table=None, add_swap=False,
                 flip_weights=None):
        """Initialize Tableflipper.

        Args:
            sublattices (list of Sublattice):
                list of Sublattices to propose steps for. Must
                be all sublattices, including active and inactive
                ones.
            flip_table(List of Dict), optional:
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
                If not given, will auto compute minimal charge neutral
                flip table.
            add_swap(Boolean, optional):
                Whether or not to attempt canonical swap if a step
                can not be proposed. May help accelerating equilibration.
                Default to False.
            flip_weights(1D Arraylike|Nonetype), optional:
                Weights to adjust probability of each flip. If
                None given, will assign equal weights to each
                flip.

        Note: In table flip, adding proposal bias to different sublattices
        is impossible.
        """
        self.all_sublattices = all_sublattices
        self.sublattices = [s for s in all_sublattices
                            if len(s.site_space) > 1]
        self.bits = [sl.species for sl in self.all_sublattices]
        self.sl_list = [list(sl.active_sites) for sl in self.all_sublattices]
        self.sl_sizes = [len(sl.sites) for sl in self.all_sublattices]
        self.sc_size = GCD_list(self.sl_sizes)
        self.sl_sizes = [s // self.sc_size for s in self.sl_sizes]

        if flip_table is None:
            self.flip_table = CompSpace(self.bits,
                                        self.sl_sizes).min_flip_table
        else:
            self.flip_table = flip_table

        self.add_swap = add_swap
        # A flip has two directions. All links initialized with 0.

        if flip_weights is None:
            self.flip_weights = np.ones(len(self.flip_table), dtype=np.int64)
        else:
            if len(flip_weights) != len(self.flip_table):
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
             list(tuple): list of tuples each with (idex, code).
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

            if len(pickable_sites) < n_picks:
                flip_list = []
                break

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
                # Site selection not successful.
                if (picked_table != picked_flip['from'][sl_id] and
                        picked_table != picked_flip['to'][sl_id]):
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
        if len(flip_list) == 0 and self.add_swap:
            active_sublattices = [sl for sl in self.sublattices
                                  if not np.all(occupancy[sl.sites] ==
                                                occupancy[sl.sites][0])]
            if len(active_sublattices) != 0:
                _swapper = Swapper(active_sublattices)
                flip_list = _swapper.propose_step(occupancy)

        return flip_list


class Subchainwalker(MCMCUsher):
    """Subchain walker class.

    Walk the unconstrained whole space with a sub-monte-carlo. Calculate
    acceptance and rejection with a bias term only to drive it back to
    constrained hyperplane, and triggers a step proposal when the bias is
    zero again, or chain length exceeds upper limit.
    """

    # Update this and MCMCKernel after you enable more biasing terms.
    valid_bias = {'null': 'Nullbias',
                  'square-charge': 'Squarechargebias'}

    def __init__(self, all_sublattices, sub_bias_type='null',
                 cutoff_steps=200, add_swap=False,
                 *args, **kwargs):
        """Initialize Subchainwalker.

        Args:
            all_sublattices(List[Sublattice]):
                List of sublattices to walk on. Must contain all sublatices,
                either active or not, otherwise may not be able to calculate
                some bias, such as charge bias.
            sub_bias_type(str):
                Type of bias term in subchain. Optional, default is null bias.
            cutoff_steps(int):
                Maximum number of subchain length. Optional, default is 200.
            add_swap(Boolean, optional):
                Whether or not to attempt canonical swap if a step
                can not be proposed. May help accelerating equilibration.
                Default to False.
            *args:
                Positional arguments to initialize bias term.
            **kwargs:
                Keyword arguments to initialize bias term.
        """
        # Keep active sublattices for perturbation only
        self.all_sublattices = all_sublattices
        self.sublattices = [s for s in all_sublattices
                            if len(s.site_space) > 1]
        self.cutoff = cutoff_steps
        self.add_swap = add_swap
        try:
            self._bias = mcbias_factory(self.valid_bias[sub_bias_type],
                                        all_sublattices,
                                        *args, **kwargs)
        except KeyError:
            raise ValueError(f"Step type {bias_type} is not valid for a "
                             f"{type(self)}.")

        self._flipper = Flipper(self.sublattices)

    def _count_species(self, occupancy):
        """Count number of species on each sublattice.

        Used to check composition equivalence.
        """
        n_cols = max(len(s.site_space) for s in self.all_sublattices)
        n_rows = len(self.all_sublattices)
        count = np.zeros((n_rows, n_cols))

        for i, s in enumerate(self.all_sublattices):
            for j, sp in enumerate(s.site_space):
                count[i, j] = np.sum(occupancy[s.sites] == j)

        return count

    def propose_step(self, occupancy):
        """Propose a step by walking subchain.

        Walk unconstrained space with constraint related penalty, and returns
        a step when constraint is satisfied again.

        Args:
            occupancy (ndarray):
                Encoded occupancy string. Does not check constraint, you have
                to check it yourself!

        Returns:
             list(tuple): list of tuples each with (idex, code).
        """
        n_attempt = 0
        step = []
        subchain_occu = occupancy.copy()

        counts_init = self._count_species(occupancy)
        counts = counts_init.copy()
        sublatt_ids = np.zeros(len(occupancy), dtype=np.int64)
        for i, s in enumerate(self.all_sublattices):
            sublatt_ids[s.sites] = i

        while (n_attempt < self.cutoff and
               (np.allclose(counts, counts_init) or
                self._bias.compute_bias(subchain_occu) != 0)):

            flip = self._flipper.propose_step(subchain_occu)
            delta_bias = self._bias.compute_bias_change(subchain_occu, flip)
            accept = (True if delta_bias <= 0
                      else -delta_bias > np.log(random.random()))
            if accept:
                i, sp = flip[0]
                counts[sublatt_ids[i], sp] += 1
                counts[sublatt_ids[i], subchain_occu[i]] -= 1

                subchain_occu[i] = sp
                step = step + flip

            n_attempt += 1

        if n_attempt >= self.cutoff:
            step = []
            if self.add_swap:
                active_sublattices = [s for s in self.sublattices
                                      if not np.all(occupancy[s.sites] ==
                                                    occupancy[s.sites][0])]

                if len(active_sublattices) != 0:
                    _swapper = Swapper(active_sublattices)
                    step = _swapper.propose_step(occupancy)

        # Clean up
        step_clean = []
        for i, sp in step:
            flipped_ids = (list(zip(*step_clean))[0] if len(step_clean) > 0
                           else [])
            if i not in flipped_ids:
                step_clean.append((i, sp))
            else:
                iid = flipped_ids.index(i)
                step_clean[iid] = (i, sp)

        return step_clean


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
