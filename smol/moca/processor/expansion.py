"""Implementation of CE processor class for a fixed size super cell.

If you are using a Hamiltonian with an Ewald summation electrostatic term, you
should use the CompositeProcessor with a ClusterExpansionProcessor and an
EwaldProcessor class to handle changes in the electrostatic interaction energy.
"""

__author__ = "Luis Barroso-Luque"

import numpy as np
from collections import defaultdict
from smol.cofe import ClusterSubspace
from smol.moca.processor.base import Processor
from src.cemc_utils import (corr_from_occupancy, factors_from_occupancy,
                            delta_corr_single_flip, delta_factors_single_flip,
                            indicator_delta_corr_single_flip)


class ClusterExpansionProcessor(Processor):
    """ClusterExpansionProcessor class to use a ClusterExpansion in MC simulations.

    A CE processor is optimized to compute correlation vectors and local
    changes in correlation vectors. This class allows the use a cluster
    expansion Hamiltonian to run Monte Carlo based simulations.

    A processor allows an ensemble class to generate a Markov chain
    for sampling thermodynamic properties from a cluster expansion
    Hamiltonian.

    Attributes:
        coefs (ndarray):
            Fitted coefficients from the cluster expansion.
        n_corr_functions (int):
            Total number of orbit basis functions (correlation functions).
            This includes all possible labellings/orderings for all orbits.
            Same as :code:`ClusterSubspace.n_bit_orderings`.
    """

    def __init__(self, cluster_subspace, supercell_matrix, coefficients):
        """Initialize a ClusterExpansionProcessor.

        Args:
            cluster_subspace (ClusterSubspace):
                A cluster subspace
            supercell_matrix (ndarray):
                An array representing the supercell matrix with respect to the
                Cluster Expansion prim structure.
            coefficients (ndarray):
                Fit coefficients for the represented cluster expansion.
        """
        super().__init__(cluster_subspace, supercell_matrix, coefficients)

        self.n_corr_functions = self.cluster_subspace.num_corr_functions
        if len(coefficients) != self.n_corr_functions:
            raise ValueError(
                f'The provided coeffiecients are not the right length. '
                f'Got {len(coefficients)} coefficients, the length must be '
                f'{self.n_corr_functions} based on the provided cluster '
                f'subspace.'
            )

        # List of orbit information and supercell site indices to compute corr
        self._orbit_list = []
        # Dictionary of orbits by site index and information
        # necessary to compute local changes in correlation vectors from flips
        self._orbits_by_sites = defaultdict(list)
        # Store the orbits grouped by site index in the structure,
        # to be used by delta_corr. We also store a reduced index array,
        # where only the rows with the site index are stored. The ratio is
        # needed because the correlations are averages over the full inds
        # array.
        # Prepare necssary information for local updates
        mappings = self._subspace.supercell_orbit_mappings(supercell_matrix)
        for cluster_indices, orbit in zip(mappings, self._subspace.orbits):
            self._orbit_list.append(
                (orbit.bit_id, orbit.flat_tensor_indices,
                 orbit.flat_correlation_tensors, cluster_indices)
            )
            for site_ind in np.unique(cluster_indices):
                in_inds = np.any(cluster_indices == site_ind, axis=-1)
                ratio = len(cluster_indices) / np.sum(in_inds)
                self._orbits_by_sites[site_ind].append(
                    (orbit.bit_id, ratio, orbit.flat_tensor_indices,
                     orbit.flat_correlation_tensors,
                     cluster_indices[in_inds])
                )

    def compute_feature_vector(self, occupancy):
        """Compute the correlation vector for a given occupancy string.

        The correlation vector in this case is normalized per supercell. In
        other works it is extensive corresponding to the size of the supercell.

        Args:
            occupancy (ndarray):
                encoded occupation array

        Returns:
            array: correlation vector
        """
        return corr_from_occupancy(occupancy, self.n_corr_functions,
                                   self._orbit_list) * self.size

    def compute_feature_vector_change(self, occupancy, flips):
        """
        Compute the change in the correlation vector from a list of flips.

        The correlation vector in this case is normalized per supercell. In
        other works it is extensive corresponding to the size of the supercell.

        Args:
            occupancy (ndarray):
                encoded occupancy string
            flips (list of tuple):
                list of tuples with two elements. Each tuple represents a
                single flip where the first element is the index of the site
                in the occupancy string and the second element is the index
                for the new species to place at that site.

        Returns:
            array: change in correlation vector
        """
        occu_i = occupancy
        delta_corr = np.zeros(self.n_corr_functions)
        for f in flips:
            occu_f = occu_i.copy()
            occu_f[f[0]] = f[1]
            site_orbit_list = self._orbits_by_sites[f[0]]
            delta_corr += self._dcorr_single_flip(
                occu_f, occu_i, self.n_corr_functions, site_orbit_list
            )
            occu_i = occu_f

        return delta_corr * self.size

    @classmethod
    def from_dict(cls, d):
        """Create a ClusterExpansionProcessor from serialized MSONable dict."""
        return cls(ClusterSubspace.from_dict(d['cluster_subspace']),
                   np.array(d['supercell_matrix']),
                   coefficients=np.array(d['coefficients']))
