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
import warnings
import math
from copy import deepcopy

from smol.utils import derived_class_factory
from .bias import mcbias_factory
from ..comp_space import (flip_vecs_to_flip_table, flip_table_to_flip_vecs,
                          CompSpace)

from ..utils.occu_utils import (occu_to_species_list, flip_weights_mask,
                                occu_to_species_stat,
                                delta_ucoords_from_step)
from ..utils.math_utils import (choose_section_from_partition, GCD_list,
                                combinatorial_number)


class MCMCUsher(ABC):
    """Abstract base class for MCMC usher classes."""

    def __init__(self, sublattices, sublattice_probabilities=None, *args,
                 **kwargs):
        """Initialize MCMCStep.

        Args:
            sublattices (list of Sublattice):
                list of Sublattices to propose steps for.
            sublattice_probabilities (list of float): optional
                list of probability to pick a site from a specific sublattice.
                Must set to 0 for inactive sublattices.
        """
        self.sublattices = sublattices
        if sublattice_probabilities is None:
            n_active = np.sum([len(s.active_sites) > 0
                              for s in self.sublattices])
            self._sublatt_probs = [(1 / n_active
                                    if len(s.active_sites) > 0 else 0)
                                   for s in self.sublattices]  # noqa
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

    def compute_a_priori_factor(self, occupancy, step):
        """Compute a-priori weight to adjust step acceptance ratio.

        This is essential in keeping detailed balance for some particular
        metropolis proposal methods.

        Args:
            occupancy (ndarray):
                encoded occupancy string
            step (list[tuple]):
                Metropolis step. list of tuples each with (index, code).

        Returns:
            float: a-priori adjustment weight.
        """
        return 1

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

    def __init__(self, all_sublattices,
                 flip_table=None, flip_vecs=None,
                 flip_weights=None, swap_weight=0, *args, **kwargs):
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
                If you have extra composition constraint besides charge,
                you have to provide your own flip table.
            flip_vecs(np.array), optional:
                Vector form of table flips, expressed in uncontrained
                coordinates.

            You only need to provide one of flip_table and flip_vecs.
            If you provide both, please confirm that they are not
            conflicting. If they conflict, we will throw an error!
            Do not provide duplicate flips in table!

            swap_weight(float, optional):
                Percentage of canonical swaps to attempt. Should be a
                positive value, but smaller than 1.
                Default to 0.
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
        self.sl_sizes = [len(sl.sites) for sl in self.all_sublattices]
        self.sc_size = GCD_list(self.sl_sizes)
        self.sl_sizes = [s // self.sc_size for s in self.sl_sizes]

        self._compspace = CompSpace(self.bits, self.sl_sizes)
        if flip_table is None and flip_vecs is None:
            self.flip_table = self._compspace.min_flip_table
            self.flip_vecs = self._compspace.unit_basis
        elif flip_table is None and flip_vecs is not None:
            self.flip_table = flip_vecs_to_flip_table(self._compspace.
                                                      unit_n_excitations,
                                                      self._compspace.
                                                      n_bits,
                                                      flip_vecs)
            self.flip_vecs = np.array(flip_vecs)
        elif flip_table is not None and flip_vecs is None:
            self.flip_table = flip_table
            self.flip_vecs = flip_table_to_flip_vecs(self._compspace.
                                                     n_bits,
                                                     flip_table)

        else:
            self.flip_table = flip_table
            flip_vec_check = flip_table_to_flip_vecs(self._compspace.
                                                     n_bits,
                                                     flip_table)
            if not np.allclose(flip_vec_check, flip_vecs):
                raise ValueError("Provided flip table and vecs can't match!")
            self.flip_vecs = np.array(flip_vecs)

        self.swap_weight = swap_weight

        # A flip has two directions.
        if flip_weights is None:
            self.flip_weights = np.ones(len(self.flip_table) * 2)
        else:
            if len(flip_weights) != len(self.flip_table):
                raise ValueError("Flip reweighted, but not enough weights" +
                                 " supplied!")
            # Forward and backward directions are assigned equal weights.
            self.flip_weights = np.repeat(flip_weights, 2)

        self._swapper = Swapper(self.sublattices, *args, **kwargs)

    def propose_step(self, occupancy):
        """Propose a single random flip step.

        A step is given as a sequence of tuples, where each tuple is of the
        form (site index, species code to set)

        1, Pick a flip in table;
        2, Choose sites from site specie statistics list according to the flip.
        3, (Acceptance proability will be adjusted according to a-priori
           weight to enforce detailed balance.)
        4, If selection failed, may or may not attempt a canonical swap based
           on self.add_swap parameter.

        There are circumstances where the current occupancy can not be accessed
        by minimal table flips. This method does not take care of that, please
        be careful when choosing starting compositions.

        Args:
            occupancy (ndarray):
                encoded occupancy string.

        Returns:
             list(tuple): list of tuples each with (idex, code).
        """
        occupancy = np.array(occupancy)
        # We shall only flip active sites.
        comp_stat = occu_to_species_stat(occupancy, self.all_sublattices,
                                         active_only=True)
        mask = flip_weights_mask(self.flip_table, comp_stat)
        masked_weights = self.flip_weights * mask
        # Masking effect will also be considered in a_priori factors.

        if random.random() < self.swap_weight:
            return self._swapper.propose_step(occupancy)
        elif masked_weights.sum() == 0:
            warnings.warn("Current occupancy can not be applied " +
                          "any table flip!")
            return self._swapper.propose_step(occupancy)
        else:
            species_list = occu_to_species_list(occupancy,
                                                self.all_sublattices,
                                                active_only=True)

            idx = choose_section_from_partition(masked_weights)
            flip = deepcopy(self.flip_table[idx // 2])

            step = []
            # Forward or backward
            if idx % 2 == 1:
                buf = deepcopy(flip['to'])
                flip['to'] = flip['from']
                flip['from'] = buf

            for sl_id in flip['from']:
                site_ids = []
                for sp_id in flip['from'][sl_id]:
                    site_ids.extend(random.sample(
                                    species_list[sl_id][sp_id],
                                    flip['from'][sl_id][sp_id]))
                for sp_id in flip['to'][sl_id]:
                    n_to = flip['to'][sl_id][sp_id]
                    chosen_ids = random.sample(site_ids, n_to)
                    for sid in chosen_ids:
                        step.append((sid, sp_id))
                        site_ids.remove(sid)

                if len(site_ids) != 0:
                    raise ValueError("Flip id {} is not num conserved!"
                                     .format(idx // 2))
            return step

    def _get_flip_id(self, occupancy, step):
        """Compute flip id in table from occupancy and step."""
        ducoords = delta_ucoords_from_step(occupancy,
                                           step,
                                           self._compspace,
                                           self.all_sublattices)

        # It is your responsibility not to have duplicate vectors in
        # flip table.
        if np.allclose(ducoords, 0):
            return -1, 0

        for fid, v in enumerate(self.flip_vecs):
            if np.allclose(v, ducoords):
                return fid, 0
            if np.allclose(-v, ducoords):
                return fid, 1

        return None, None

    def compute_a_priori_factor(self, occupancy, step):
        """Compute a-priori weight to adjust step acceptance ratio.

        This is essential in keeping detailed balance for some particular
        metropolis proposal methods.

        Arg:
            occupancy (ndarray):
                encoded occupancy string
            step (list[tuple]):
                Metropolis step. list of tuples each with (index, code).

        Returns:
            float: a-priori adjustment weight.
        """
        fid, direction = self._get_flip_id(occupancy, step)
        if fid is None:
            raise ValueError("Step {} is not in flip table.".format(step))

        if fid < 0:
            # Canonical swap
            return 1

        flip = deepcopy(self.flip_table[fid])

        if direction == 1:
            # Backwards direction.
            buf = deepcopy(flip['to'])
            flip['to'] = flip['from']
            flip['from'] = buf

        comp_stat_now = occu_to_species_stat(occupancy, self.all_sublattices,
                                             active_only=True)
        mask_now = flip_weights_mask(self.flip_table, comp_stat_now)
        weights_now = self.flip_weights * mask_now
        p_flip_now = ((1 - self.swap_weight) *
                      weights_now[fid * 2 + direction]
                      / weights_now.sum())

        comp_stat_next = deepcopy(comp_stat_now)
        for sl_id in flip['from']:
            for sp_id in flip['from'][sl_id]:
                comp_stat_next[sl_id][sp_id] -= flip['from'][sl_id][sp_id]
        for sl_id in flip['to']:
            for sp_id in flip['to'][sl_id]:
                comp_stat_next[sl_id][sp_id] += flip['to'][sl_id][sp_id]
        mask_next = flip_weights_mask(self.flip_table, comp_stat_next)
        weights_next = self.flip_weights * mask_next
        p_flip_next = ((1 - self.swap_weight) *
                       weights_next[fid * 2 + (1 - direction)]
                       / weights_next.sum())

        # Combinatorial factor.
        comb_factor = p_flip_next / p_flip_now
        for sl_id in flip['from']:
            factor_sl = 1
            for sp_id in flip['from'][sl_id]:
                dn = flip['from'][sl_id][sp_id]
                n_now = comp_stat_now[sl_id][sp_id]
                n_next = comp_stat_next[sl_id][sp_id]
                assert n_next == n_now - dn

                factor_sl *= (math.factorial(dn) *
                              combinatorial_number(n_now, dn))

            for sp_id in flip['to'][sl_id]:
                dn = flip['to'][sl_id][sp_id]
                n_now = comp_stat_now[sl_id][sp_id]
                n_next = comp_stat_next[sl_id][sp_id]
                assert n_next == n_now + dn

                factor_sl /= (math.factorial(dn) *
                              combinatorial_number(n_next, dn))

            comb_factor *= factor_sl

        return comb_factor


class Subchainwalker(MCMCUsher):
    """Subchain walker class.

    Walk the unconstrained whole space with a sub-monte-carlo. Calculate
    acceptance and rejection with a bias term only to drive it back to
    constrained hyperplane, and triggers a step proposal when the bias is
    zero again, or chain length exceeds upper limit.
    """

    # Update this and MCMCKernel after you enable more biasing terms.
    valid_bias = {'null': 'Nullbias',
                  'square-charge': 'Squarechargebias',
                  'square-comp-constraint': 'Squarecompconstraintbias'}

    def __init__(self, all_sublattices, sub_bias_type='null',
                 cutoff_steps=200, minimize_swap=False, add_swap=True,
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
            minimize_swap(Boolean, optional):
                If true, will try to minimize number of swaps by requiring the
                subchain to terminate at not only bias=0, but also change of
                composition!=0. Default is False, because allowing more swaps
                helps incresing acceptace, and therefore faster equilibration.
            add_swap(Boolean, optional):
                Whether or not to attempt canonical swap if a step
                can not be proposed. Helps accelerating equilibration and
                sampling.
                Default to True.
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
        self.minimize_swap = minimize_swap
        self.add_swap = add_swap
        try:
            self._bias = mcbias_factory(self.valid_bias[sub_bias_type],
                                        all_sublattices,
                                        *args, **kwargs)
        except KeyError:
            raise ValueError(f"Bias type {sub_bias_type} is not valid for a "
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

        while n_attempt < self.cutoff:
            if (self._bias.compute_bias(subchain_occu) == 0 and
                    (not np.allclose(subchain_occu, occupancy))):
                if self.minimize_swap:
                    if not np.allclose(counts, counts_init):
                        break
                else:
                    if n_attempt > 0:  # Allows swaps to appear.
                        break

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
