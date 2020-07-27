"""
Implementation of a Canonical Ensemble Class.

Used when running Monte Carlo simulations for fixed number of sites and fixed
concentration of species.
"""

__author__ = "Luis Barroso-Luque"

from smol.moca.ensemble.base import Ensemble
from smol.moca.processor.base import Processor


class CanonicalEnsemble(Ensemble):
    """
    A Canonical Ensemble class to run Monte Carlo Simulations.
    """
    valid_move_types = ('swap',)

    def __init__(self, processor, temperature, sublattices=None,
                 sublattice_probabilities=None, move_type=None):
        """Initialize CanonicalEnemble.

        Args:
            processor (Processor):
                A processor that can compute the change in a property given
                a set of flips.
            temperature (float):
                Temperature of ensemble
            sublattices (list of Sublattice): optional
                list of Lattice objects representing sites in the processor
                supercell with same site spaces.
            sublattice_probabilities (list of float): optional
                list of probability to pick a site from a specific sublattice.
            move_type (str):
                string specifying the type of MCMC move for the Canonical
                ensemble.
        """
        # TODO check for site space overlap and warn
        if move_type is None:
            move_type = 'swap'
        elif move_type not in self.valid_move_types:
            raise ValueError(f'Provided move type {move_type} is not a valid '
                             'option for a Canonical Ensemble. Valid options '
                             f'are {self.valid_move_types}.')
        super().__init__(processor, temperature, move_type=move_type,
                         sublattices=sublattices,
                         sublattice_probabilities=sublattice_probabilities)

    @property
    def exponential_parameters(self):
        """Get the vector of exponential parameters."""
        return self.processor.coefs

    def compute_sufficient_statistics(self, occupancy):
        """Compute the sufficient statistics for a give occupancy.

        In the canonical case it is just the feature vector.

        Args:
            occupancy (ndarray):
                encoded occupancy string

        Returns:
            ndarray: vector of sufficient statistics
        """
        return -self.beta * self.processor.compute_feature_vector(occupancy)

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
        return -self.beta * self.processor.compute_feature_vector_change(occupancy, move)  # noqa
