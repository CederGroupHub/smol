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
    valid_mcmc_ushers = ('Swapper',)

    def __init__(self, processor, temperature, sublattices=None):
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
        """
        super().__init__(processor, temperature, sublattices=sublattices)

    @property
    def natural_parameters(self):
        """Get the vector of exponential parameters."""
        return self.processor.coefs

    def compute_feature_vector(self, occupancy):
        """Compute the feature vector for a give occupancy


        In the canonical case it is just the feature vector.

        Args:
            occupancy (ndarray):
                encoded occupancy string

        Returns:
            ndarray: feature vector
        """
        return self.processor.compute_feature_vector(occupancy)

    def compute_feature_vector_change(self, occupancy, step):
        """Compute the change in the feature vector from a step.

        Args:
            occupancy (ndarray):
                encoded occupancy string.
            step (list of tuple):
                A sequence of flips as given my the MCMCUsher.propose_step

        Returns:
            ndarray: difference in feature vector
        """
        return self.processor.compute_feature_vector_change(occupancy, step)
