"""Implementation of CE processor class for a fixed size supercell.

If you are using a Hamiltonian with an Ewald summation electrostatic term, you
should use the CompositeProcessor with a ClusterExpansionProcessor and an
EwaldProcessor class to handle changes in the electrostatic interaction energy.
"""

__author__ = "Luis Barroso-Luque"

from collections import defaultdict

import numpy as np

from smol.cofe.space.clusterspace import ClusterSubspace
from smol.correlations import corr_from_occupancy, delta_corr_single_flip
from smol.moca.processor.base import Processor


class ClusterExpansionProcessor(Processor):
    """ClusterExpansionProcessor class to use a ClusterExpansion in MC.

    A CE processor is optimized to compute correlation vectors and local
    changes in correlation vectors. This class allows the use of a CE
    Hamiltonian to run Monte Carlo-based simulations

    A processor that allows an Ensemble class to generate a Markov chain
    for sampling thermodynamic properties from a CE Hamiltonian.

    Attributes:
        coefs (ndarray):
            Fitted coefficients from the cluster expansion.
        num_corr_functions (int):
            Total number of orbit basis functions (correlation functions).
            This includes all possible labelings/orderings for all orbits.
            Same as :code:`ClusterSubspace.n_bit_orderings`.
    """

    def __init__(self, cluster_subspace, supercell_matrix, coefficients):
        """Initialize a ClusterExpansionProcessor.

        Args:
            cluster_subspace (ClusterSubspace):
                a cluster subspace
            supercell_matrix (ndarray):
                an array representing the supercell matrix with respect to the
                Cluster Expansion prim structure.
            coefficients (ndarray):
                fit coefficients for the represented cluster expansion.
        """
        super().__init__(cluster_subspace, supercell_matrix, coefficients)

        self.num_corr_functions = self.cluster_subspace.num_corr_functions
        if len(coefficients) != self.num_corr_functions:
            raise ValueError(
                f"The provided coefficients are not the right length. "
                f"Got {len(coefficients)} coefficients, the length must be "
                f"{self.num_corr_functions} based on the provided cluster "
                f"subspace."
            )

        # List of orbit information and supercell site indices to compute corr
        self._orbit_list = cluster_subspace.gen_orbit_list(supercell_matrix)

        # Dictionary of orbits by site index and information
        # necessary to compute local changes in correlation vectors from flips
        self._orbits_by_sites = defaultdict(list)
        # Store the orbits grouped by site index in the structure,
        # to be used by delta_corr. We also store a reduced index array,
        # where only the rows with the site index are stored. The ratio is
        # needed because the correlations are averages over the full inds
        # array.
        for (
            bit_id,
            flat_tensor_inds,
            flat_corr_tensors,
            cluster_indices,
        ) in self._orbit_list:
            for site_ind in np.unique(cluster_indices):
                in_inds = np.any(cluster_indices == site_ind, axis=-1)
                ratio = len(cluster_indices) / np.sum(in_inds)
                self._orbits_by_sites[site_ind].append(
                    (
                        bit_id,
                        ratio,
                        flat_tensor_inds,
                        flat_corr_tensors,
                        cluster_indices[in_inds],
                    )
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
        return (
            corr_from_occupancy(occupancy, self.num_corr_functions, self._orbit_list)
            * self.size
        )

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
        delta_corr = np.zeros(self.num_corr_functions)
        for f in flips:
            occu_f = occu_i.copy()
            occu_f[f[0]] = f[1]
            site_orbit_list = self._orbits_by_sites[f[0]]
            delta_corr += delta_corr_single_flip(
                occu_f, occu_i, self.num_corr_functions, site_orbit_list
            )
            occu_i = occu_f

        return delta_corr * self.size

    @classmethod
    def from_dict(cls, d):
        """Create a ClusterExpansionProcessor from serialized MSONable dict."""
        return cls(
            ClusterSubspace.from_dict(d["cluster_subspace"]),
            np.array(d["supercell_matrix"]),
            coefficients=np.array(d["coefficients"]),
        )
