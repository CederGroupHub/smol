"""Implementation of target feature processors.

Feature distance processors are used to compute a distance between the feature values
of a particular configuration and a target feature vector. This functionality is useful
to obtain special structures, such as SQS, SSOS, etc.
"""

__author__ = "Luis Barroso-Luque"

from abc import ABCMeta, abstractmethod
from itertools import chain

import numpy as np

from smol.cofe.space.clusterspace import ClusterSubspace
from smol.moca.processor import ClusterDecompositionProcessor, ClusterExpansionProcessor
from smol.moca.processor.base import Processor

# TODO implement this with evaluator
corr_distance_single_flip = None
interaction_distance_single_flip = None


class DistanceProcessor(Processor, metaclass=ABCMeta):
    """Abstract class to compute distance from a fixed feature vector.

    The distance used to measure distance is,

    .. math::

        d = -wL + ||W^T(f - f_T)||_1

    where f is the feature vector, f_T is the target feature vector, and L is the
    diameter for which all values of clusters of smaller diameter are exactly
    equal to the target, and w is a weight for the L term. And W is a diagonal matrix
    of optional weights for each feature.

    As proposed in the following publication,
    https://doi.org/10.1016/j.calphad.2013.06.006
    """

    def __init__(
        self,
        cluster_subspace,
        supercell_matrix,
        target_vector,
        match_weight=1.0,
        match_tol=1e-5,
        target_weights=None,
        **processor_kwargs,
    ):
        """Initialize a DistanceProcessor.

        Args:
            cluster_subspace (ClusterSubspace):
                a cluster subspace
            supercell_matrix (ndarray):
                an array representing the supercell matrix with respect to the
                Cluster Expansion prim structure.
            target_vector (ndarray): optional
                target feature vector (excluding constant)
            match_weight (float): optional
                weight for the in the wL term above. That is how much to weight the
                largest diameter below which all features are matched exactly.
                Set to any number >0 to use this term. Default is 1.0. to ignore it
                set it to zero.
            match_tol (float): optional
                tolerance for matching features. Default is 1e-5.
            target_weights (ndarray): optional
                Weights for the absolute differences each feature when calculating
                the total distance. If None, then all features are weighted equally.
            processor_kwargs (dict): optional
                additional keyword arguments to instantiate the underlying processor
                being inherited from.
        """
        self.target_vector = target_vector

        if match_weight < 0:
            raise ValueError("The match weight must be a positive number.")
        self.match_tol = match_tol

        if len(target_weights) != len(target_vector) - 1:
            raise ValueError(
                f"The length of target_weights must be equal to the length of"
                f"the target vector minus one {len(target_vector) - 1}. \n"
                f"Got {len(target_weights)} instead."
            )

        super().__init__(
            cluster_subspace,
            supercell_matrix,
            coefficients=np.concatenate([[-match_weight], target_weights]),
            **processor_kwargs,
        )

    @abstractmethod
    def compute_feature_vector_distances(self, occupancy, flips):
        """Compute distance of feature vectors separated by given flips from the target.

        Args:
            occupancy (ndarray):
                encoded occupancy array
            flips (list of tuple):
                list of tuples with two elements. Each tuple represents a
                single flip where the first element is the index of the site
                in the occupancy array and the second element is the index
                for the new species to place at that site.

        Returns:
            ndarray: 2D distance vector where the first row corresponds to correlations
            before flips
        """

    @abstractmethod
    def exact_match_max_diameter(self, distance_vector):
        """
        Get the largest diameter for which all features are exactly matched.

        Args:
            distance_vector (ndarray):
                feature vector

        Returns:
            float: largest diameter for which all features are exactly matched.
        """
        return

    def compute_feature_vector(self, occupancy):
        """
        Compute the change in the feature vector from a list of flips.

        Args:
            occupancy (ndarray):
                encoded occupancy array
            flips (list of tuple):
                list of tuples with two elements. Each tuple represents a
                single flip where the first element is the index of the site
                in the occupancy array and the second element is the index
                for the new species to place at that site.

        Returns:
            array: change in feature vector
        """
        feature_vector = (
            super().compute_feature_vector(occupancy) / self.size
        )  # remove scaling
        feature_vector[1:] = np.abs(feature_vector[1:] - self.target_vector[1:])

        if self.coefs[0] > 0:
            feature_vector[0] = self.exact_match_max_diameter(feature_vector)
        else:
            feature_vector[0] = 0.0

        return feature_vector

    def compute_feature_vector_change(self, occupancy, flips):
        """
        Compute the change in the feature vector from a list of flips.

        The feature vector in this case is normalized per supercell. In
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
            array: change in feature vector
        """
        distance_vectors = self.compute_feature_vector_distances(occupancy, flips)

        if self.coefs[0] > 0:
            distance_vectors[0] = self.exact_match_max_diameter(distance_vectors[0])
            distance_vectors[1] = self.exact_match_max_diameter(distance_vectors[1])

        return distance_vectors[1] - distance_vectors[0]

    def as_dict(self):
        """Return a dictionary representation of the processor."""
        d = super().as_dict()
        d["target_vector"] = self.target_vector.tolist()
        d["match_weight"] = -self.coefs[0]
        return d

    @classmethod
    def from_dict(cls, d):
        """Create a processor from a dictionary representation."""
        cluster_subspace = ClusterSubspace.from_dict(d.pop("cluster_subspace"))
        supercell_matrix = np.array(d.pop("supercell_matrix"))
        target_vector = np.array(d.pop("target_vector"))
        match_weight = d.pop("match_weight")

        # remove @ and coefficient values:
        d = {k: v for k, v in d.items() if "@" not in k and k != "coefficients"}
        return cls(
            cluster_subspace,
            supercell_matrix=supercell_matrix,
            target_vector=target_vector,
            match_weight=match_weight,
            **d,
        )


class CorrelationDistanceProcessor(DistanceProcessor, ClusterExpansionProcessor):
    """CorrelationDistanceProcessor to compute distance from a fixed correlation vector.

    The distance used to measure distance is,

    .. math::

        d = -wL + ||W^T(f - f_T)||_1

    where f is the correlation vector, f_T is the target correlation vector, and L is the
    diameter for which all correlation values of clusters of smaller diameter are exactly
    equal to the target, and w is a weight for the L term. And W is a diagonal matrix
    of optional weights for each correlation.

    As proposed in the following publication,
    https://doi.org/10.1016/j.calphad.2013.06.006
    """

    def __init__(
        self,
        cluster_subspace,
        supercell_matrix,
        target_vector=None,
        match_weight=1.0,
        target_weights=None,
        match_tol=1e-5,
    ):
        """Initialize a CorrelationDistanceProcessor.

        Args:
            cluster_subspace (ClusterSubspace):
                a cluster subspace
            supercell_matrix (ndarray):
                an array representing the supercell matrix with respect to the
                Cluster Expansion prim structure.
            target_vector (ndarray): optional
                target correlation vector, if None given as vector of zeros is used.
            match_weight (float): optional
                weight for the in the wL term above. That is how much to weight the
                largest diameter below which all correlations are matched exactly.
                Set to any number >0 to use this term. Default is 1.0. to ignore it
                set it to zero.
            match_tol (float): optional
                tolerance for matching features. Default is 1e-8.
            target_weights (ndarray): optional
                Weights for the absolute differences each correlation when calculating
                the total distance. If None, then all correlations are weighted equally.
        """
        # TODO do not accept external terms here....
        if target_vector is None:
            target_vector = np.zeros(len(cluster_subspace))
        # TODO check length and raise error if incorrect

        if target_weights is None:
            target_weights = np.ones(len(cluster_subspace) - 1)

        super().__init__(
            cluster_subspace,
            supercell_matrix,
            target_vector=target_vector,
            match_weight=match_weight,
            target_weights=target_weights,
            match_tol=match_tol,
        )

    def compute_feature_vector_distances(self, occupancy, flips):
        """Compute the distance of correlation vectors separated by given flips.

        The distance is referenced to the target correlation vector, such that the
        result is a 2D array where each row is the distance of one correlation vector
        to the target.

        Args:
            occupancy (ndarray):
                encoded occupancy array
            flips (list of tuple):
                list of tuples with two elements. Each tuple represents a single flip
                where the first element is the index of the site in the occupancy array
                and the second element is the index for the new species to place at that
                site.

        Returns:
            ndarray: 2D distance vector where the first row corresponds to correlations
            before flips
        """
        occu_i = occupancy
        corr_distances = np.zeros((2, self.cluster_subspace.num_corr_functions))
        for f in flips:
            occu_f = occu_i.copy()
            occu_f[f[0]] = f[1]
            corr_distances += self._evaluator.delta_corr_distance_from_occupancies(
                occu_f, occu_i, self.target_vector, self._indices.container
            )
            occu_i = occu_f

        return corr_distances

    def exact_match_max_diameter(self, distance_vector):
        """
        Get the largest diameter for which all features are exactly matched.

        Args:
            distance_vector (ndarray):
                feature vector

        Returns:
            float: largest diameter
        """
        max_matched_diameter = 0.0

        for diameter, orbits in self.cluster_subspace.orbits_by_diameter.items():
            indices = list(
                chain.from_iterable(
                    range(orb.bit_id, orb.bit_id + len(orb)) for orb in orbits
                )
            )
            if np.all(distance_vector[indices] <= self.match_tol):
                max_matched_diameter = diameter
            else:
                break

        return max_matched_diameter


class ClusterInteractionDistanceProcessor(
    DistanceProcessor, ClusterDecompositionProcessor
):
    """Compute distances from a fixed cluster interaction vector.

    The distance used to measure distance is,

    .. math::

        d = -wL + ||W^T(f - f_T)||_1

    where f is the cluster interaction vector, f_T is the target vector, and L is the
    diameter for which all correlation values of clusters of smaller diameter are
    exactly equal to the target, and w is a weight for the L term. And W is a diagonal
    matrix of optional weights for each correlation.

    Following the proposal in the following publication but applied to cluster
    interaction vectors,
    https://doi.org/10.1016/j.calphad.2013.06.006
    """

    def __init__(
        self,
        cluster_subspace,
        supercell_matrix,
        interaction_tensors,
        target_vector=None,
        match_weight=1.0,
        target_weights=None,
        match_tol=1e-5,
    ):
        """Initialize a CorrelationDistanceProcessor.

        Args:
            cluster_subspace (ClusterSubspace):
                a cluster subspace
            supercell_matrix (ndarray):
                an array representing the supercell matrix with respect to the
                Cluster Expansion prim structure.
            interaction_tensors (sequence of ndarray):
                Sequence of ndarray where each array corresponds to the
                cluster interaction tensor. These should be in the same order as their
                corresponding orbits in the given cluster_subspace
            target_vector (ndarray): optional
                target cluster interaction vector, if None given as vector of zeros is

                used.
            match_weight (float): optional
                weight for the in the wL term above. That is how much to weight the
                largest diameter below which all correlations are matched exactly.
                Set to any number >0 to use this term. Default is 1.0. to ignore it
                set it to zero.
            match_tol (float): optional
                tolerance for matching features. Default is 1e-8.
            target_weights (ndarray): optional
                Weights for the absolute differences each correlation when calculating
                the total distance. If None, then all correlations are weighted equally.
        """
        if target_vector is None:
            target_vector = np.zeros(cluster_subspace.num_orbits)

        if target_weights is None:
            target_weights = np.ones(cluster_subspace.num_orbits - 1)

        interaction_tensors = tuple(
            np.array(interaction) for interaction in interaction_tensors
        )

        super().__init__(
            cluster_subspace,
            supercell_matrix,
            interaction_tensors=interaction_tensors,
            target_vector=target_vector,
            match_weight=match_weight,
            target_weights=target_weights,
            match_tol=match_tol,
        )

    def compute_feature_vector_distances(self, occupancy, flips):
        """Compute the distance of cluster interaction vectors separated by given flips.

        Args:
            occupancy (ndarray):
                encoded occupancy array
            flips (list of tuple):
                list of tuples with two elements. Each tuple represents a single flip
                where the first element is the index of the site in the occupancy array
                and the second element is the index for the new species to place at that
                site.

        Returns:
            ndarray: 2D distance vector where the first row corresponds to correlations
            before flips
        """
        occu_i = occupancy
        interaction_distances = np.zeros((2, self._subspace.num_orbits))
        for f in flips:
            occu_f = occu_i.copy()
            occu_f[f[0]] = f[1]
            interaction_distances += (
                self._evaluator.delta_interaction_distance_from_occupancies(
                    occu_f, occu_i, self.target_vector, self._indices.container
                )
            )
            occu_i = occu_f

        return interaction_distances

    def exact_match_max_diameter(self, distance_vector):
        """
        Get the largest diameter for which all features are exactly matched.

        Args:
            distance_vector (ndarray):
                feature vector

        Returns:
            float: largest diameter
        """
        max_matched_diameter = 0.0

        for diameter, orbits in self.cluster_subspace.orbits_by_diameter.items():
            indices = [orb.id for orb in orbits]
            if np.all(distance_vector[indices] <= self.match_tol):
                max_matched_diameter = diameter
            else:
                break

        return max_matched_diameter
