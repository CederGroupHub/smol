"""Implementation of CE processor class for a fixed size supercell.

If you are using a Hamiltonian with an Ewald summation electrostatic term, you
should use the CompositeProcessor with a ClusterExpansionProcessor and an
EwaldProcessor class to handle changes in the electrostatic interaction energy.
"""

__author__ = "Luis Barroso-Luque"

from collections import defaultdict

import numpy as np

from smol.cofe.space.clusterspace import ClusterSubspace
from smol.correlations import (
    corr_from_occupancy,
    delta_corr_single_flip,
    delta_interactions_single_flip,
    interactions_from_occupancy,
)
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


class ClusterDecompositionProcessor(Processor):
    """ClusterDecompositionProcessor to use a CD in MC simulations.

    An ClusterDecompositionProcessor is similar to a ClusterExpansionProcessor
    however it computes properties and changes of properties using cluster interactions
    rather than correlation functions and ECI.

    For a given ClusterExpansion the results from sampling using this class
    should be the exact same as using a ClusterExpansionProcessor (ie the
    samples are from the same ensemble).

    A few practical and important differences though, the featyre vectors
    that are used and sampled are the cluster interactions for each occupancy
    as opposed to the correlation vectors (so if you need correlation vectors
    for further analysis you're probably better off with a standard
    ClusterExpansionProcessor

    However if you don't need correlation vectors, then this is probably
    more efficient and more importantly the scaling complexity with model size
    is in terms of number of orbits (instead of number of corr functions).
    This can give substantial speedups when doing MC calculations of complex
    high component systems.
    """

    def __init__(self, cluster_subspace, supercell_matrix, interaction_tensors):
        """Initialize an ClusterDecompositionProcessor.

        Args:
            cluster_subspace (ClusterSubspace):
                A cluster subspace
            supercell_matrix (ndarray):
                An array representing the supercell matrix with respect to the
                Cluster Expansion prim structure.
            interaction_tensors (sequence of ndarray):
                Sequence of ndarray where each array corresponds to the
                cluster interaction tensor. These should be in the same order as their
                corresponding orbits in the given cluster_subspace
        """
        if len(interaction_tensors) != cluster_subspace.num_orbits:
            raise ValueError(
                f"The number of cluster interaction tensors must match the number "
                f" of orbits in the subspace. Got {len(interaction_tensors)} "
                f"interaction tensors, but need {cluster_subspace.num_orbits} "
                f" for the given cluster_subspace."
            )

        #  the orbit multiplicities play the role of coefficients here.
        super().__init__(
            cluster_subspace,
            supercell_matrix,
            coefficients=cluster_subspace.orbit_multiplicities,
        )

        self.n_orbits = self.cluster_subspace.num_orbits
        self._fac_tensors = interaction_tensors

        # List of orbit information and supercell site indices to compute corr
        self._orbit_list = []
        # Dictionary of orbits by site index and information
        # necessary to compute local changes in correlation vectors from flips
        self._orbits_by_sites = defaultdict(list)
        # Prepare necessary information for local updates
        mappings = self._subspace.supercell_orbit_mappings(supercell_matrix)
        for cluster_indices, interaction_tensor, orbit in zip(
            mappings, self._fac_tensors[1:], self._subspace.orbits
        ):
            flat_interaction_tensor = np.ravel(interaction_tensor, order="C")
            self._orbit_list.append(
                (orbit.flat_tensor_indices, flat_interaction_tensor, cluster_indices)
            )
            for site_ind in np.unique(cluster_indices):
                in_inds = np.any(cluster_indices == site_ind, axis=-1)
                ratio = len(cluster_indices) / np.sum(in_inds)
                self._orbits_by_sites[site_ind].append(
                    (
                        orbit.id,
                        ratio,
                        orbit.flat_tensor_indices,
                        flat_interaction_tensor,
                        cluster_indices[in_inds],
                    )
                )

    def compute_feature_vector(self, occupancy):
        """Compute the cluster interaction vector for a given occupancy string.

        The cluster interaction vector in this case is normalized per supercell.

        Args:
            occupancy (ndarray):
                encoded occupation array

        Returns:
            array: correlation vector
        """
        return (
            interactions_from_occupancy(
                occupancy, self.n_orbits, self._fac_tensors[0], self._orbit_list
            )
            * self.size
        )

    def compute_feature_vector_change(self, occupancy, flips):
        """
        Compute the change in the cluster interaction vector from a list of flips.

        The cluster interaction vector in this case is normalized per supercell.

        Args:
            occupancy (ndarray):
                encoded occupancy string
            flips (list of tuple):
                list of tuples with two elements. Each tuple represents a
                single flip where the first element is the index of the site
                in the occupancy string and the second element is the index
                for the new species to place at that site.

        Returns:
            array: change in cluster interaction vector
        """
        occu_i = occupancy
        delta_interactions = np.zeros(self.n_orbits)
        for f in flips:
            occu_f = occu_i.copy()
            occu_f[f[0]] = f[1]
            site_orbit_list = self._orbits_by_sites[f[0]]
            delta_interactions += delta_interactions_single_flip(
                occu_f, occu_i, self.n_orbits, site_orbit_list
            )
            occu_i = occu_f

        return delta_interactions * self.size

    def as_dict(self) -> dict:
        """
        Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        d = super().as_dict()
        d["interaction_tensors"] = [
            interaction.tolist() for interaction in self._fac_tensors
        ]
        return d

    @classmethod
    def from_dict(cls, d):
        """Create a ClusterExpansionProcessor from serialized MSONable dict."""
        interaction_tensors = tuple(
            np.array(interaction) for interaction in d["interaction_tensors"]
        )
        return cls(
            ClusterSubspace.from_dict(d["cluster_subspace"]),
            np.array(d["supercell_matrix"]),
            interaction_tensors,
        )
