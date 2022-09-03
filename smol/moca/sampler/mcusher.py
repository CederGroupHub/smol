"""Implementations of MCMC Usher classes.

An usher is used to generate step proposals for MC Monte Carlo sampling.
For example a Flipper simply proposes a change of the identity of a species
at a site for use in a SemiGrand ensemble. A Swapper will propose a swap
between species at two sites for use in Canonical ensemble simulations.

More complex steps can be defined simply by deriving from the MCUsher.
"""

__author__ = "Luis Barroso-Luque"

from abc import ABC, abstractmethod

import numpy as np

from smol.utils import class_name_from_str, derived_class_factory


class MCUsher(ABC):
    """Abstract base class for MC usher classes."""

    def __init__(self, sublattices, sublattice_probabilities=None, rng=None):
        """Initialize MCMCStep.

        Args:
            sublattices (list of Sublattice):
                list of active Sublattices to propose steps for. Active
                sublattices are those that include sites with configuration
                degrees of freedom DOFs, only occupancy on active sub-lattices'
                active sites are allowed to change.
            sublattice_probabilities (list of float): optional
                list of probabilities to pick a site from specific active
                sublattices.
            rng (np.Generator): optional
                The given PRNG must be the same instance as that used by the kernel and
                any bias terms, otherwise reproducibility will be compromised.
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
                "Sublattice probabilites needs to be the same length as sublattices."
            )
        elif sum(sublattice_probabilities) != 1:
            raise ValueError("Sublattice probabilites must sum to one.")
        else:
            self._sublatt_probs = sublattice_probabilities

        self._rng = np.random.default_rng(rng)

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
            list(tuple): tuple of tuples each with (index, code)
        """
        return []

    def update_aux_state(self, step, *args, **kwargs):
        """Update any auxiliary state information based on an accepted step."""

    def set_aux_state(self, state, *args, **kwargs):
        """Set the auxiliary state from a checkpoint value."""

    def get_random_sublattice(self):
        """Return a random sublattice based on given probabilities."""
        return self._rng.choice(self.active_sublattices, p=self._sublatt_probs)


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
        sublattice = self.get_random_sublattice()
        site = self._rng.choice(sublattice.active_sites)
        choices = set(sublattice.encoding) - {occupancy[site]}
        return [(site, self._rng.choice(list(choices)))]


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
            list(tuple): list of tuples each with (index, code)
        """
        sublattice = self.get_random_sublattice()
        site1 = self._rng.choice(sublattice.active_sites)
        species1 = occupancy[site1]
        sublattice_occu = occupancy[sublattice.active_sites]
        swap_options = sublattice.active_sites[sublattice_occu != species1]
        if swap_options.size > 0:  # check if swap_options are not empty
            site2 = self._rng.choice(swap_options)
            swap = [(site1, occupancy[site2]), (site2, species1)]
        else:
            # inefficient, maybe re-call method? infinite recursion problem
            swap = []
        return swap


class MultiStep(MCUsher):
    """A multi step usher to generate steps of several flips.

    Any usher can be used to chain steps, ie create multi flip steps, multi swap steps
    even multi composite steps, etc.
    """

    def __init__(
        self,
        sublattices,
        mcusher,
        step_lengths,
        sublattice_probabilities=None,
        step_probabilities=None,
        rng=None,
    ):
        """Initialize a multistep usher.

        Args:
            sublattices (list of Sublattice):
                list of active Sublattices to propose steps for. Active
                sublattices are those that include sites with configuration
                degrees of freedom DOFs, only occupancy on active sub-lattices' active sites
                are allowed to change.
            mcusher (MCUsher):
                The MCUsher to use to create multisteps.
            step_lengths (int or Sequence of int):
                The length or lengths of steps. If given a sequence of sizes, a size
                is choosen at each proposal according to step_probabilities.
            sublattice_probabilities (list of float): optional
                list of probabilities to pick a site from specific active
                sublattices.
            step_probabilities (Sequence of float): optional
                A sequence of probabilites corresponding to the step lengths provided.
                If none is given, then a uniform probability is used.
            rng (np.Generator): optional
                The given PRNG must be the same instance as that used by the kernel and
                any bias terms, otherwise reproducibility will be compromised.
        """
        super().__init__(sublattices, sublattice_probabilities, rng=rng)

        if isinstance(step_lengths, int):
            self._step_lens = np.array([step_lengths], dtype=int)
        else:
            self._step_lens = np.array(step_lengths, dtype=int)

        if step_probabilities is not None:
            if sum(step_probabilities) != 1.0:
                raise ValueError("The step_probabilities do not sum to 1.")
            if len(step_probabilities) != len(step_lengths):
                raise ValueError(
                    "The lenght of step_lengths and step_probabilities does not match."
                )
            self.step_p = np.array(step_probabilities)
        else:
            self._step_p = np.array(
                [1.0 / len(self._step_lens) for _ in range(len(self._step_lens))]
            )

        if isinstance(mcusher, str):  # TODO additional kwargs?
            mcusher = mcusher_factory(
                mcusher,
                self.sublattices,
                sublattice_probabilities=self.sublattice_probabilities,
            )

        self._mcusher = mcusher

    def propose_step(self, occupancy):
        """Propose a step given an occupancy."""
        step_length = self._rng.choice(self._step_lens, p=self._step_p)
        steps = []
        occu = occupancy.copy()
        for _ in range(step_length):
            steps.append(self._mcusher.propose_step(occu))
            for f in steps[-1]:
                occu[f[0]] = f[1]

        # unpack them to single list
        return [flip for step in steps for flip in step]


class Composite(MCUsher):
    """A composite usher for chaining different step types.

    This can be used to mix step types for all sublattices or to create hybrid
    ensembles where different sublattices have different allowed steps.
    """

    def __init__(
        self,
        sublattices,
        sublattice_probabilities=None,
        mcushers=None,
        weights=None,
        rng=None,
    ):
        """Initialize the composite usher.

        Args:
            sublattices (list of Sublattice):
                list of active Sublattices to propose steps for. Active
                sublattices are those that include sites with configuration
                degrees of freedom DOFs, only occupancy on active sub-lattices' active sites
                are allowed to change.
            sublattice_probabilities (list of float): optional
                list of probabilities to pick a site from specific active
                sublattices.
            mcushers (list of MCUsher):
                A list of mcushers to add to the composite.
            weights (list of float):
                A list of the weights associated with each mcusher passed. With this
                the corresponding probabilities for each mcusher to be picked are
                generated. Must be in the same order as the mcusher list.
            rng (np.Generator): optional
                The given PRNG must be the same instance as that used by the kernel and
                any bias terms, otherwise reproducibility will be compromised.
        """
        super().__init__(sublattices, sublattice_probabilities, rng=rng)
        self._mcushers = []
        self._weights = []
        self._p = []

        if mcushers is not None:
            if weights is None:
                weights = len(mcushers) * [
                    1,
                ]
            for weight, usher in zip(weights, mcushers):
                if isinstance(usher, str):
                    usher = mcusher_factory(
                        usher,
                        self.sublattices,
                        sublattice_probabilities=self.sublattice_probabilities,
                    )
                self.add_mcusher(usher, weight)

    @property
    def mcushers(self):
        """Get the list of mcushers."""
        return self._mcushers

    @property
    def weight(self):
        """Get the weights associated with each mcusher."""
        return self._weights

    def add_mcusher(self, mcusher, weight=1):
        """Add an MCUsher to the Composite usher.

        Args:
            mcusher (MCUsher):
                The MCUsher to add
            weight (float):
                The weight associated with the mcusher being added
        """
        self._mcushers.append(mcusher)
        self._update_p(weight)

    def _update_p(self, weight):
        """Update the probabilities for each mcuhser based on a new weight."""
        self._weights.append(weight)
        total = sum(self._weights)
        self._p = [weight / total for weight in self._weights]

    def propose_step(self, occupancy):
        """Propose a step given an occupancy."""
        return self._rng.choice(self._mcushers, p=self._p).propose_step(occupancy)


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
            keyword arguments passed to class constructor

    Returns:
        MCUsher: instance of derived class.
    """
    usher_name = class_name_from_str(usher_type)
    return derived_class_factory(usher_name, MCUsher, sublattices, *args, **kwargs)
