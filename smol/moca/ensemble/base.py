"""Abstract Base class for Monte Carlo Ensembles."""

__author__ = "Luis Barroso-Luque"

from abc import ABC, ABCMeta, abstractmethod
import numpy as np

from smol.constants import kB
from .sublattice import Sublattice
from .moves import mcmc_move_factory


class Ensemble(ABC):
    """Abstract Base Class for Monte Carlo Ensembles."""

    def __init__(self, processor, temperature, move_type, sublattices=None,
                 sublattice_probabilities=None):
        """Initialize class instance.

        Args:
            processor (Processor):
                A processor that can compute the change in a property given
                a set of flips.
            move_type (str):
                string specifying the type of MCMC move for the ensemble.
            sublattices (list of Sublattice): optional
                list of Lattice objects representing sites in the processor
                supercell with same site spaces.
            sublattice_probabilities (list of float): optional
                list of probability to pick a site from a specific sublattice.
        """
        if sublattices is None:
            self._sublattices = [Sublattice(site_space,
                                            np.array([i for i, sp in
                                              enumerate(processor.allowed_species)  # noqa
                                              if sp == list(site_space.keys())]))  # noqa
                                 for site_space in processor.unique_site_spaces]  # noqa
            if sublattice_probabilities is None:
                self._sublatt_prob = len(self._sublattices) * [1/len(self._sublattices), ]  # noqa
            elif len(sublattice_probabilities) != len(self._sublattices):
                raise AttributeError('Sublattice probabilites needs to be the '
                                     'same length as sublattices.')
            else:
                self.sublatattice_probabilities = sublattice_probabilities
        else:
            self._sublattices = sublattices

        self.temperature = temperature
        self._processor = processor
        self._sublattices = sublattices
        self.restricted_sites = []
        self._move = mcmc_move_factory(move_type, sublattices,
                                       sublattice_probabilities)
        # save this for resetting purposes.
        self.__move_type = move_type
        self.__sublat_probs = sublattice_probabilities

    @property
    def temperature(self):
        """Get the temperature of ensemble."""
        return self.__temperature

    @temperature.setter
    def temperature(self, temperature):
        """Set the temperature and beta accordingly."""
        self.__temperature = temperature
        self.__beta = 1.0 / (kB * temperature)

    @property
    def beta(self):
        """Get 1/kBT."""
        return self.__beta

    @property
    def num_sites(self):
        """Get the total number of atoms in supercell."""
        return self.processor.num_sites

    @property
    def system_size(self):
        """Get size of supercell in number of prims."""
        return self.processor.size

    @property
    def processor(self):
        """Get the system processor."""
        return self._processor

    # TODO make a setter for this that checks sublattices are correct and
    #  all sites are included.
    @property
    def sublattices(self):
        """Get names of sublattices.

        Useful if allowing flips only from certain sublattices is needed.
        """
        return self._sublattices

    @property
    def active_sublattices(self):
        """Get the active sublattices."""
        return self._move.sublattices

    @property
    def active_sublattice_probabilities(self):
        """Get the active sublattice probabilities."""
        return self._move.sublattice_probabilities

    @active_sublattice_probabilities.setter
    def active_sublattice_probabilities(self, value):
        """Set the active sublattice probabilities."""
        self._move.sublattice_probabilities = value

    @property
    @abstractmethod
    def exponential_parameters(self):
        """Get the vector of exponential parameters.

        The exponential parameters correspond to the fit coeficients of the
        underlying processor plus any additional terms involved in the Legendre
        transformation corresponding to the ensemble.
        """
        return

    @abstractmethod
    def compute_sufficient_statistics(self, occupancy):
        """Compute the sufficient statistics for a give occupancy

        The vector of sufficient statistics is the the necessary to compute
        the exponent determining in the relative probability for the given
        occupancy (i.e. The LT for a generalized Massieu function).

        Args:
            occupancy (ndarray):
                encoded occupancy string

        Returns:
            ndarray: vector of sufficient statistics
        """
        return

    @abstractmethod
    def compute_sufficient_statistics_change(self, occupancy, move):
        """Return the change in the sufficient statistics vector from a move.

        Args:
            occupancy (ndarray):
                encoded occupancy string.
            move (list of tuple):
                A sequence of moves given my the MCMCMove.propose.

        Returns:
            ndarray: difference in vector of sufficient statistics
        """
        return

    def propose_mcmc_move(self, occupancy):
        """Return a MCMC move.

        Args:
            occupancy (ndarray):

        Returns:
            list of tuple: a move where each tuple is (site, species code)
        """
        return self._move.propose(occupancy)

    def restrict_sublattice(self, sublattice):
        """Restrict a whole sublattice."""
        sublattices = [sublatt for sublatt in self.sublattices
                       if sublatt is not sublattice]
        probabilities = [prob for prob in self.__sublat_probs]
        probabilities = [prob/sum(probabilities) for prob in probabilities]
        self._move = mcmc_move_factory(self.__move_type, sublattices,
                                       probabilities)

    def reset_restricted_sublattices(self):
        """Reset the activate sublattices to all lattices."""
        self._move = mcmc_move_factory(self.__move_type, self.sublattices,
                                       self.__sublat_probs)

    def restrict_sites(self, sites):
        """Restricts (freezes) the given sites.

        This will exclude those sites from being flipped during a Monte Carlo
        run. If some of the given indices refer to inactive sites, there will
        be no effect.

        Args:
            sites (Sequence):
                indices of sites in the occupancy string to restrict.
        """
        for sublattice in self.sublattices:
            sublattice.restrict_sites(sites)

    def reset_restricted_sites(self):
        """Unfreeze all previously restricted sites."""
        for sublattice in self.sublattices:
            sublattice.reset_restricted_sites()

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
           tuple: (minimum energy, occupancy, annealing data)
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

        anneal_data = {}
        for T in temperatures:
            self.temperature = T
            self.run(mc_iterations)
            anneal_data[T] = self.data
            self._data = []

        min_occupancy = self.processor.decode_occupancy(min_occupancy)
        return min_energy, min_occupancy, anneal_data
