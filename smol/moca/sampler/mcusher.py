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

from smol.utils import derived_class_factory, class_name_from_str
from ..comp_space import CompSpace

from ..utils.occu_utils import (get_dim_ids_by_sublattice,
                                get_dim_ids_table,
                                occu_to_species_list,
                                occu_to_species_n,
                                delta_n_from_step)
from ..utils.math_utils import (choose_section_from_partition, gcd_list,
                                comb, flip_weights_mask)


# TODO  keep RNG as attribute for reproducibility, pass as optional constructor...
class MCUsher(ABC):
    """Abstract base class for MC usher classes."""

    def __init__(self, sublattices, sublattice_probabilities=None):
        """Initialize MCMCStep.

        Args:
            sublattices (list of Sublattice):
                list of active Sublattices to propose steps for. Active
                sublattices are those that include sites with configuration
                DOFs, only occupancy on active sub-lattices' active sites
                are allowed to change.
            sublattice_probabilities (list of float): optional
                list of probability to pick a site from a specific active
                sublattice.
        """
        self.sublattices = sublattices
        self.active_sublattices = [
            sublatt for sublatt in self.sublattices if sublatt.is_active
        ]

        if sublattice_probabilities is None:
            self._sublatt_probs = np.array(
                len(self.active_sublattices)
                * [
                    1 / len(self.active_sublattices),
                ]
            )
        elif len(sublattice_probabilities) != len(self.active_sublattices):
            raise AttributeError(
                "Sublattice probabilites needs to be the " "same length as sublattices."
            )
        elif sum(sublattice_probabilities) != 1:
            raise ValueError("Sublattice probabilites must sum to one.")
        else:
            self._sublatt_probs = sublattice_probabilities

    @property
    def sublattice_probabilities(self):
        """Get the sublattice probabilities."""
        return self._sublatt_probs

    @sublattice_probabilities.setter
    def sublattice_probabilities(self, value):
        """Set the sublattice probabilities."""
        if len(value) != len(self.active_sublattices):
            raise AttributeError(
                f"Can not set sublattice probabilities.\n Length must be the"
                f" same as the number of sublattices {len(self.sublattices)}"
            )
        if sum(value) != 1:
            raise ValueError(
                "Can not set sublattice probabilities.\n"
                "Sublattice probabilites must sum to one."
            )
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

    def compute_log_priori_factor(self, occupancy, step):
        """Compute a-priori weight to adjust step acceptance ratio.

        This is essential in keeping detailed balance for some particular
        metropolis proposal methods. Return log of the factor for numerical
        accuracy.

        Args:
            occupancy (ndarray):
                encoded occupancy string
            step (list[tuple]):
                Metropolis step. list of tuples each with (index, code).

        Returns:
            float: log of a-priori adjustment weight.
        """
        return 0

    def update_aux_state(self, step, *args, **kwargs):
        """Update any auxiliary state information based on an accepted step."""

    def set_aux_state(self, state, *args, **kwargs):
        """Set the auxiliary state from a checkpoint values."""

    def get_random_sublattice(self):
        """Return a random sublattice based on given probabilities."""
        rng = np.random.default_rng()
        return rng.choice(self.active_sublattices, p=self._sublatt_probs)


class Flip(MCUsher):
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
        rng = np.random.default_rng()
        sublattice = self.get_random_sublattice()
        site = rng.choice(sublattice.active_sites)
        choices = set(sublattice.encoding) - {occupancy[site]}
        return [(site, rng.choice(list(choices)))]


class Swap(MCUsher):
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
        rng = np.random.default_rng()
        sublattice = self.get_random_sublattice()
        site1 = rng.choice(sublattice.active_sites)
        species1 = occupancy[site1]
        sublattice_occu = occupancy[sublattice.active_sites]
        swap_options = sublattice.active_sites[sublattice_occu != species1]
        if swap_options.size > 0:  # check if swap_options are not empty
            site2 = rng.choice(swap_options)
            swap = [(site1, occupancy[site2]), (site2, species1)]
        else:
            # inefficient, maybe re-call method? infinite recursion problem
            swap = []
        return swap


class Tableflip(MCUsher):
    """Implementation of table flips.

    Use user-specified or auto-calculated flip tables. Does not
    check the correctness of the flip tables.

    No longer computing and assigning a-priori proabilities
    before flip selection, and selects sites instead.

    Direct use of this class is not recommended, because the
    dict form of the flip table is not very easy to read and
    write.
    """

    def __init__(self, sublattices, flip_table=None,
                 charge_balanced=True,
                 other_constraints=None,
                 optimize_basis=False,
                 table_ergodic=False,
                 flip_weights=None, swap_weight=0):
        """Initialize Tableflip.

        Args:
            sublattices (list of Sublattice):
                list of Sublattices to propose steps for. Must
                be all sublattices, including active and inactive
                ones.
            flip_table(2D np.ndarray[int]), optional:
                Flip vectors in "n" format, as generated by a CompSpace.
                If not given, we will initialize a CompSpace from
                sublattice informations, and calculate the flip table
                with that CompSpace.
                Notice:
                1, We suggest you provide a pre-computed flip table, unless
                the number of sites in sub-lattices can be reduced to very
                small when divided with their GCD.
                2, As implemented in the Sublattice class,
                you are allowed to have multiple species on an inactive
                sub-lattice, but it is not the suggested approach.
                In this case, you have 3 choices. Either always try to split
                until all inactive sub-lattices only have one species
                (suggested), fixing number of inactive species as a
                constraint, or pre-compute your own flip table.
            charge_balanced(bool): optional
                Whether to add charge balance constraint. Default
                to true.
            other_constraints(List[tuple(1D arrayLike[int], int)]): optional
                Other integer constraints except charge balance and
                site-number conservation. Should be given in the form of
                tuple(a, bb), each gives constraint np.dot(a, n)=bb.
            optimize_basis(bool): optional
                Whether to optimize the basis to minimal flip sizes and maximal
                connectivity in the minimum super-cell size.
                When the minimal super-cell size is large, we recommend not to
                optimize basis.
            table_ergodic(bool): optional
                When generating a flip table, whether to add vectors and
                ensure ergodicity under a minimal super-cell size.
                Default to False.
                When the minimal super-cell size is large, we recommend not to
                ensure ergodicity. This is not only because of the computation
                difficulty; but also because at large super-cell size,
                the fraction of inaccessible compositions usually becomes
                minimal.
            swap_weight(float): optional
                Percentage of canonical swaps to attempt. Should be a
                positive value, but smaller than 1.
                Default to 0.
            flip_weights(1D Arraylike): optional
                Weights to adjust probability of each flip. If
                None given, will assign equal weights to each
                flip vector and its inverse. Length must equal to
                2*len(flip_table).

        Note: In table flip, adding proposal bias to different
        sub-lattices is not meaningful. We will always assign
        sub-lattice probabilties as equal.
        """
        super(Tableflip, self).__init__(sublattices)
        self.bits = [sl.species for sl in self.sublattices]
        self.dim_ids = get_dim_ids_by_sublattice(self.bits)
        self.sl_sizes = np.array([len(sl.sites) for sl in self.sublattices],
                                 dtype=int)
        self.sc_size = gcd_list(self.sl_sizes)
        self.sl_sizes = self.sl_sizes // self.sc_size
        self.max_n = [len(sublatt.active_sites)
                      for sublatt in self.sublattices]

        self._compspace = CompSpace(self.bits, self.sl_sizes,
                                    charge_balanced=charge_balanced,
                                    other_constraints=other_constraints,
                                    optimize_basis=optimize_basis,
                                    table_ergodic=table_ergodic)
        if flip_table is not None:
            self.flip_table = np.array(flip_table, dtype=int)
        else:
            self.flip_table = self._compspace.flip_table

        self.swap_weight = swap_weight

        # A flip has two directions.
        if flip_weights is None:
            self.flip_weights = np.ones(len(self.flip_table) * 2)
        else:
            if (len(flip_weights) != len(self.flip_table)
                    and len(flip_weights) != len(self.flip_table) * 2):
                raise ValueError(f"{len(flip_weights)} weights provided. "
                                 "You must provide either 1* or 2* weights "
                                 f"given {len(self.flip_table)} flip "
                                 "vectors!")
            # Forward and backward directions are assigned equal weights.
            if len(flip_weights) == len(self.flip_table):
                self.flip_weights = np.repeat(flip_weights, 2)
            if len(flip_weights) == 2 * len(self.flip_table):
                self.flip_weights = np.array(flip_weights)

        self._swapper = Swap(self.sublattices)
        self._dim_ids_table = get_dim_ids_table(self.sublattices,
                                                active_only=True)

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
        rng = np.random.default_rng()
        if rng.random() < self.swap_weight:
            return self._swapper.propose_step(occupancy)

        # We shall only flip active sites.
        species_list = occu_to_species_list(occupancy, self._dim_ids_table)
        species_n = [len(sites) for sites in species_list]
        mask = flip_weights_mask(self.flip_table,
                                 species_n, self.max_n).astyle(int)
        masked_weights = self.flip_weights * mask
        # Mask out impossible selections.
        if masked_weights.sum() == 0:
            warnings.warn("Current occupancy is not ergodic! "
                          "Will do canonical swap only!")
            return self._swapper.propose_step(occupancy)

        idx = choose_section_from_partition(masked_weights)
        u = self.flip_table[idx // 2]
        # Forward or backward
        if idx % 2 == 1:
            u = -1 * u

        # Sub-lattices can not cross.
        step = []
        for sl_id, (sublatt, dim_ids) \
                in enumerate(zip(self.sublattices, self.dim_ids)):
            if not sublatt.is_active:
                continue
            site_ids = []
            dim_ids = np.array(dim_ids, type=int)
            u_sl = u[dim_ids]
            dims_from = dim_ids[u_sl < 0]
            dims_to = dim_ids[u_sl > 0]
            codes_to = sublatt.encoding[u_sl > 0]
            for d in dims_from:
                site_ids.extend(rng.choice(species_list[d],
                                           size=-1 * u[d],
                                           replace=False).tolist())
            for d, code in zip(dims_to, codes_to):
                for site_id in rng.choice(site_ids, size=u[d],
                                          replace=False):
                    step.append((int(site_id), int(code)))
                    site_ids.remove(site_id)
            assert len(site_ids) == 0  # Site num conservation.

        return step

    def _get_flip_id(self, occupancy, step):
        """Compute flip id in table from occupancy and step."""
        dn = delta_n_from_step(occupancy, step, self._dim_ids_table)

        if np.allclose(dn, 0):
            return -1, 0

        for fid, v in enumerate(self.flip_table):
            if np.allclose(v, dn):
                return fid, 0
            if np.allclose(-v, dn):
                return fid, 1

        return None, None

    def compute_log_priori_factor(self, occupancy, step):
        """Compute a-priori weight to adjust step acceptance ratio.

        This is essential in keeping detailed balance. Return log for
        numerical accuracy.

        Arg:
            occupancy (ndarray):
                encoded occupancy string. (Processor encoding.)
            step (list[tuple]):
                Metropolis step. list of tuples each with (index, code).
                (Processor encoding.)

        Returns:
            float: log of a-priori adjustment ratio.
        """
        fid, direction = self._get_flip_id(occupancy, step)
        if fid is None:
            raise ValueError(f"Step {step} is not in flip table.")

        if fid < 0:
            # Canonical swap
            return 1

        u = (-2 * direction + 1) * self.flip_table[fid]

        n_now = occu_to_species_n(occupancy, self._dim_ids_table)
        mask_now = flip_weights_mask(self.flip_table,
                                     n_now, self.max_n).astype(int)
        weights_now = self.flip_weights * mask_now
        p_now = ((1 - self.swap_weight) * weights_now[fid * 2 + direction]
                 / weights_now.sum())

        n_next = n_now + u
        mask_next = flip_weights_mask(self.flip_table,
                                      n_next, self.max_n).astyle(int)
        weights_next = self.flip_weights * mask_next
        p_next = ((1 - self.swap_weight)
                  * weights_next[fid * 2 + (1 - direction)]
                  / weights_next.sum())

        # Combinatorial factor.
        log_factor = np.log(p_next / p_now)
        dim_ids = np.arange(len(u), type=int)
        dims_from = dim_ids[u < 0]
        dims_to = dim_ids[u > 0]

        for dim in dims_from:
            log_factor += np.log(math.factorial(-u[dim])
                                 * comb(n_now[dim], -u[dim]))
        for dim in dims_to:
            log_factor -= np.log(math.factorial(u[dim])
                                 * comb(n_next[dim], u[dim]))

        return log_factor


def mcusher_factory(usher_type, sublattices, *args, **kwargs):
    """Get a MC Usher from string name.

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
        MCUsher: instance of derived class.
    """
    usher_name = class_name_from_str(usher_type)
    return derived_class_factory(usher_name, MCUsher, sublattices, *args, **kwargs)
