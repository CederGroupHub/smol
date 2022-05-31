"""
Implementation of a Canonical Ensemble class.

These are used when running Monte Carlo simulations for systems
with a fixed number of sites and fixed concentration of species.
"""

__author__ = "Luis Barroso-Luque"


from monty.json import MSONable

from smol.moca.ensemble.base import BaseEnsemble
from smol.moca.processor.base import Processor
from smol.moca.sublattice import Sublattice


class CanonicalEnsemble(BaseEnsemble, MSONable):
    """Canonical Ensemble class to run Monte Carlo Simulations."""

    @property
    def natural_parameters(self):
        """Get the vector of exponential parameters."""
        return self.processor.coefs

    def compute_feature_vector(self, occupancy):
        """Compute the feature vector for a given occupancy.

        In the canonical case it is just the feature vector from the underlying
        processor.

        Args:
            occupancy (ndarray):
                encoded occupancy string

        Returns:
            ndarray: feature vector
        """
        return self.processor.compute_feature_vector(occupancy)

    def compute_feature_vector_change(self, occupancy, step):
        """Compute the change in the feature vector from a given step.

        Args:
            occupancy (ndarray):
                encoded occupancy string.
            step (list of tuple):
                A sequence of flips as given by MCUsher.propose_step

        Returns:
            ndarray: difference in feature vector
        """
        return self.processor.compute_feature_vector_change(occupancy, step)

    @classmethod
    def from_dict(cls, d):
        """Instantiate a CanonicalEnsemble from dict representation.

        Args:
            d (dict):
                dictionary representation.
        Returns:
            CanonicalEnsemble
        """
        return cls(
            Processor.from_dict(d["processor"]),
            [Sublattice.from_dict(s) for s in d["sublattices"]],
        )
