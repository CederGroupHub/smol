"""Implementation of target feature processors.

Feature distance processors are used to compute a distance between the feature values
of a particular configuration and a target feature vector. This functionality is useful
to obtain special structures, such as SQS, SSOS, etc.
"""

__author__ = "Luis Barroso-Luque"

from abc import ABCMeta, abstractmethod

import numpy as np

from smol.cofe.space.clusterspace import ClusterSubspace
from smol.correlations import delta_corr_dist_single_flip
from smol.moca.processor import ClusterExpansionProcessor
from smol.moca.processor.base import Processor


class FeatureDistanceProcessor(Processor, metaclass=ABCMeta):
    """FeaturedistanceProcessor abstract class to compute distance from a fixed feature vector.

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
        target_features,
        match_weight=None,
        target_weights=None,
        **processor_kwargs
    ):
        """Initialize a TargetfeatureProcessor.

        Args:
            cluster_subspace (ClusterSubspace):
                a cluster subspace
            supercell_matrix (ndarray):
                an array representing the supercell matrix with respect to the
                Cluster Expansion prim structure.
            target_features (ndarray): optional
                target feature vector
            match_weight (float): optional
                weight for the in the wL term above. That is how much to weight the
                largest diameter below which all features are matched exactly.
                Set to any number >0 to use this term. If None, then the term is
                ignored.
            target_weights (ndarray): optional
                Weights for the absolute differences each feature when calculating
                the total distance. If None, then all features are weighted equally.
            processor_kwargs (dict): optional
                additional keyword arguments to instantiate the underlying processor
                being inherited from.
        """
        self.target = target_features
        if match_weight is not None:
            if match_weight <= 0:
                raise ValueError("The match weight must be a positive number.")
        self.match_weight = match_weight

        super().__init__(
            cluster_subspace,
            supercell_matrix,
            coefficients=target_weights,
            **processor_kwargs
        )

    @abstractmethod
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
        return np.abs(super().compute_feature_vector(occupancy) - self.target)

    def compute_property(self, occupancy):
        """Compute the distance of the feature vector from the target.

        Args:
            occupancy (ndarray):
                encoded occupancy array
        Returns:
            float: distance from target
        """
        distance = super().compute_property(occupancy)
        # TODO now need to find the largest diameter with distance == 0 for -wL term
        if self.match_weight is not None:
            pass

        return distance

    def compute_property_change(self, occupancy, flips):
        """Compute the change in distance of the feature vector from the target.

        Args:
            occupancy (ndarray):
                encoded occupancy array
            flips (list of tuple):
                list of tuples with two elements. Each tuple represents a
                single flip where the first element is the index of the site
                in the occupancy array and the second element is the index
                for the new species to place at that site.

        Returns:
            float: change in distance from target
        """
        distance_diff = super().compute_property_change(occupancy, flips)
        # TODO need to find the largest diameter with distance == 0 same as above
        if self.match_weight is not None:
            pass

        return distance_diff

    def as_dict(self):
        """Return a dictionary representation of the processor."""
        d = super().as_dict()
        d["target"] = self.target.tolist()
        d["match_weight"] = self.match_weight
        return d

    @classmethod
    def from_dict(cls, d):
        """Create a processor from a dictionary representation."""
        cluster_subspace = ClusterSubspace.from_dict(d.pop("cluster_subspace"))
        supercell_matrix = np.array(d.pop("supercell_matrix"))
        target = np.array(d.pop("target"))
        match_weight = d.pop("match_weight")

        return cls(cluster_subspace, supercell_matrix, target, match_weight, **d)


class CorrelationDistanceProcessor(FeatureDistanceProcessor, ClusterExpansionProcessor):
    """CorrelationDistanceProcessor to compute distances from a fixed correlation vector.

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
        target_correlations=None,
        match_weight=None,
        target_weights=None,
    ):
        """Initialize a CorrelationDistanceProcessor.

        Args:
            cluster_subspace (ClusterSubspace):
                a cluster subspace
            supercell_matrix (ndarray):
                an array representing the supercell matrix with respect to the
                Cluster Expansion prim structure.
            target_correlations (ndarray): optional
                target correaltion vector, if None given as vector of zeros is used.
            match_weight (float): optional
                weight for the in the wL term above. That is how much to weight the
                largest diameter below which all correlations are matched exactly.
                Set to any number >0 to use this term. If None, then the term is
                ignored.
            target_weights (ndarray): optional
                Weights for the absolute differences each correlation when calculating
                the total distance. If None, then all correlations are weighted equally.
        """
        if target_correlations is None:
            target_correlations = np.zeros(len(cluster_subspace))

        if target_weights is None:
            target_weights = np.ones(len(cluster_subspace))

        super().__init__(
            cluster_subspace,
            supercell_matrix,
            target_features=target_correlations,
            match_weight=match_weight,
            target_weights=target_weights,
        )

    def compute_feature_vector_change(self, occupancy, flips):
        """
        Compute the change of correlation vector differences from a fixed vector from a list of flips.

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
        occu_i = occupancy
        delta_corr = np.zeros(self.num_corr_functions)
        for f in flips:
            occu_f = occu_i.copy()
            occu_f[f[0]] = f[1]
            delta_corr += delta_corr_dist_single_flip(
                occu_f,
                occu_i,
                self.target,
                self.num_corr_functions,
                self._orbit_list,
            )
            occu_i = occu_f

        return delta_corr * self.size
