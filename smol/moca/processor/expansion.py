"""Implementation of CE processor class for a fixed size supercell.

If you are using a Hamiltonian with an Ewald summation electrostatic term, you
should use the CompositeProcessor with a ClusterExpansionProcessor and an
EwaldProcessor class to handle changes in the electrostatic interaction energy.
"""

__author__ = "Luis Barroso-Luque"

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from smol.cofe.space.clusterspace import ClusterSubspace, OrbitIndices
from smol.moca.processor.base import Processor
from smol.utils.cluster import get_orbit_data
from smol.utils.cluster.container import IntArray2DContainer
from smol.utils.cluster.evaluator import ClusterSpaceEvaluator
from smol.utils.cluster.numthreads import SetNumThreads
from smol.utils.setmany import SetMany


@dataclass
class LocalEvalData:
    """Holds data for local evaluation updates.

    A dataclass to the data for local evaluation updates of correlation and cluster
    interaction vectors.
    """

    site_index: int
    evaluator: ClusterSpaceEvaluator = field(repr=False)
    indices: OrbitIndices = field(repr=False)
    cluster_ratio: np.ndarray
    num_threads: int = field(default=SetNumThreads("evaluator"), repr=True)


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

    num_threads = SetMany("num_threads", "_eval_data_by_sites")
    num_threads_full = SetNumThreads("_evaluator")

    def __init__(
        self,
        cluster_subspace,
        supercell_matrix,
        coefficients,
        num_threads=None,
        num_threads_full=None,
    ):
        """Initialize a ClusterExpansionProcessor.

        Args:
            cluster_subspace (ClusterSubspace):
                a cluster subspace
            supercell_matrix (ndarray):
                an array representing the supercell matrix with respect to the
                Cluster Expansion prim structure.
            coefficients (ndarray):
                fit coefficients for the represented cluster expansion.
            num_threads (int): optional
                Number of threads to use to compute `updates in a correlation vector
                from local changes. If None is given the number of threads are set
                to the same as set in the given :class:`ClusterSubspace`.
                Note that this is not saved when serializing the ClusterSubspace with
                the as_dict method, so if you are loading a ClusterSubspace from a
                file then make sure to set the number of threads as desired.
            num_threads_full (int): optional
                Number of threads to use to compute a full correlation vector. Note
                that this is not saved when serializing the ClusterSubspace with the
                as_dict method, so if you are loading a ClusterSubspace from a
                file then make sure to set the number of threads as desired.
        """
        super().__init__(cluster_subspace, supercell_matrix, coefficients)

        if len(coefficients) != self.cluster_subspace.num_corr_functions:
            raise ValueError(
                f"The provided coefficients are not the right length. "
                f"Got {len(coefficients)} coefficients, the length must be "
                f"{self.cluster_subspace.num_corr_functions} based on the provided cluster "
                f"subspace."
            )

        # evaluator for the correlation functions
        self._evaluator = ClusterSpaceEvaluator(
            get_orbit_data(cluster_subspace.orbits),
            cluster_subspace.num_orbits,
            cluster_subspace.num_corr_functions,
        )

        # orbit indices mapping all sites in this supercell to clusters in each
        # orbit. This is used to compute the correlation vector.
        self._indices = cluster_subspace.get_orbit_indices(supercell_matrix)

        # TODO there is a lot of repeated code here and in the CD __init__
        # TODO we should probably refactor this to avoid this and make testing easier
        # Dictionary of orbits data by site index necessary to compute local changes in
        # correlation vectors from flips
        data_by_sites = defaultdict(list)
        # We also store a reduced cluster index array where only the rows with the site
        # index are stored. The ratio is also needed because the correlations are
        # averages over the full cluster indices array.
        for orbit, cluster_indices in zip(
            cluster_subspace.orbits, self._indices.arrays
        ):
            orbit_data = (
                orbit.id,
                orbit.bit_id,
                orbit.flat_correlation_tensors,
                orbit.flat_tensor_indices,
            )
            for site_ind in np.unique(cluster_indices):
                in_inds = np.any(cluster_indices == site_ind, axis=-1)
                ratio = len(cluster_indices) / np.sum(in_inds)
                data_by_sites[site_ind].append(
                    (orbit_data, cluster_indices[in_inds], ratio)
                )

        # now unpack the data, create the local evaluators and store them in a
        # dictionary indexed by site index.
        self._eval_data_by_sites = {}
        for site, data in data_by_sites.items():
            # we don't really need a different evaluator for each site since sym
            # equivalent sites can share the same one, same goes for ratio.
            evaluator = ClusterSpaceEvaluator(
                tuple(d[0] for d in data),
                cluster_subspace.num_orbits,
                cluster_subspace.num_corr_functions,
            )
            indices = tuple(d[1] for d in data)
            orbit_indices = OrbitIndices(indices, IntArray2DContainer(indices))
            ratio = np.array([d[2] for d in data])
            self._eval_data_by_sites[site] = LocalEvalData(
                site, evaluator, orbit_indices, ratio, 1
            )

        # Set the number of threads to use for the full correlation vector
        self.num_threads_full = (
            num_threads_full if num_threads_full else cluster_subspace.num_threads
        )
        # Set the number of threads to use for the local correlation vector
        self.num_threads = num_threads

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
            self._evaluator.correlations_from_occupancy(
                occupancy, self._indices.container
            )
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
        delta_corr = np.zeros(self.cluster_subspace.num_corr_functions)
        for f in flips:
            occu_f = occu_i.copy()
            occu_f[f[0]] = f[1]
            eval_data = self._eval_data_by_sites[f[0]]
            delta_corr += eval_data.evaluator.delta_correlations_from_occupancies(
                occu_f, occu_i, eval_data.cluster_ratio, eval_data.indices.container
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

    However, if you don't need correlation vectors, then this is probably
    more efficient and more importantly the scaling complexity with model size
    is in terms of number of orbits (instead of number of corr functions).
    This can give substantial speedups when doing MC calculations of complex
    high component systems.
    """

    num_threads = SetMany("num_threads", "_eval_data_by_sites")
    num_threads_full = SetNumThreads("_evaluator")

    def __init__(
        self,
        cluster_subspace,
        supercell_matrix,
        interaction_tensors,
        coefficients=None,
        num_threads=None,
        num_threads_full=None,
    ):
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
                corresponding orbits in the given cluster_subspace.
            num_threads (int): optional
                Number of threads to use to compute a correlation vector. Note that
                this is not saved when serializing the ClusterSubspace with the
                as_dict method, so if you are loading a ClusterSubspace from a
                file then make sure to set the number of threads as desired.
                corresponding orbits in the given cluster_subspace
            coefficients (ndarray): optional
                coefficients of each mean cluster interaction tensor. This should often
                be some multiple of the orbit multiplicities. Default is the orbit
                multiplicities.
        """
        if len(interaction_tensors) != cluster_subspace.num_orbits:
            raise ValueError(
                f"The number of cluster interaction tensors must match the number "
                f" of orbits in the subspace. Got {len(interaction_tensors)} "
                f"interaction tensors, but need {cluster_subspace.num_orbits} "
                f" for the given cluster_subspace."
            )

        #  the orbit multiplicities play the role of coefficients here.
        coefficients = (
            cluster_subspace.orbit_multiplicities
            if coefficients is None
            else coefficients
        )
        super().__init__(cluster_subspace, supercell_matrix, coefficients=coefficients)

        self._interaction_tensors = interaction_tensors  # keep these for serialization

        flat_interaction_tensors = tuple(
            np.ravel(tensor, order="C") for tensor in interaction_tensors[1:]
        )

        self._evaluator = ClusterSpaceEvaluator(
            get_orbit_data(cluster_subspace.orbits),
            cluster_subspace.num_orbits,
            cluster_subspace.num_corr_functions,
            1,  # set threads to here so we can set it later
            interaction_tensors[0],
            flat_interaction_tensors,
        )

        # orbit indices mapping all sites in this supercell to clusters in each
        # orbit. This is used to compute the correlation vector.
        self._indices = cluster_subspace.get_orbit_indices(supercell_matrix)

        # Dictionary of orbits data by site index necessary to compute local changes in
        # correlation vectors from flips
        data_by_sites = defaultdict(list)
        # We also store a reduced cluster index array where only the rows with the site
        # index are stored. The ratio is also needed because the correlations are
        # averages over the full cluster indices array.
        for orbit, cluster_indices, interaction_tensor in zip(
            cluster_subspace.orbits, self._indices.arrays, flat_interaction_tensors
        ):
            orbit_data = (
                orbit.id,
                orbit.bit_id,
                orbit.flat_correlation_tensors,
                orbit.flat_tensor_indices,
            )
            for site_ind in np.unique(cluster_indices):
                in_inds = np.any(cluster_indices == site_ind, axis=-1)
                ratio = len(cluster_indices) / np.sum(in_inds)
                data_by_sites[site_ind].append(
                    (orbit_data, cluster_indices[in_inds], ratio, interaction_tensor)
                )

        # now unpack the data, create the local evaluators and store them in a
        # dictionary indexed by site index.
        self._eval_data_by_sites = {}
        for site, data in data_by_sites.items():
            orbit_data = tuple(d[0] for d in data)
            indices = tuple(d[1] for d in data)
            orbit_indices = OrbitIndices(indices, IntArray2DContainer(indices))
            ratio = np.array([d[2] for d in data])
            flat_interaction_tensors = tuple(d[3] for d in data)
            # we don't really need a different evaluator for each site since sym
            # equivalent sites can share the same one, same goes for ratio.
            evaluator = ClusterSpaceEvaluator(
                orbit_data,
                cluster_subspace.num_orbits,
                cluster_subspace.num_corr_functions,
                1,  # set to single thread, will be set by num_threads later
                interaction_tensors[0],
                flat_interaction_tensors,
            )
            self._eval_data_by_sites[site] = LocalEvalData(
                site, evaluator, orbit_indices, ratio, 1
            )

        # Set the number of threads to use for the full correlation vector
        self.num_threads_full = (
            num_threads_full if num_threads_full else cluster_subspace.num_threads
        )
        # Set the number of threads to use for the local correlation vector
        self.num_threads = num_threads

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
            self._evaluator.interactions_from_occupancy(
                occupancy, self._indices.container
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
        delta_interactions = np.zeros(self.cluster_subspace.num_orbits)
        for f in flips:
            occu_f = occu_i.copy()
            occu_f[f[0]] = f[1]
            eval_data = self._eval_data_by_sites[f[0]]
            delta_interactions += (
                eval_data.evaluator.delta_interactions_from_occupancies(
                    occu_f, occu_i, eval_data.cluster_ratio, eval_data.indices.container
                )
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
            interaction.tolist() for interaction in self._interaction_tensors
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
