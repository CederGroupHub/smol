"""Implementation of target feature processors.

Target feature processors are used to compute a distance between the feature values
of a particular configuration and a target feature vector. This functionality is useful
to obtain special structures, such as SQS, SSOS, etc.
"""

__author__ = "Luis Barroso-Luque"

from abc import ABCMeta, abstractmethod
import numpy as np


from smol.cofe.space.clusterspace import ClusterSubspace
from smol.correlations import corr_from_occupancy, delta_corr_single_flip
from smol.moca.processor.base import Processor


class TargetFeatureProcessor(Processor, metaclass=ABCMeta):
    """TargetCorrelationProcessor class to compute distance from a correlation vector.

    """
    # make this take a cluter expansion or a decomposition processor and use that!!!
    def __init__(self, cluster_subspace, supercell_matrix, target_features=None,
                 feature_weights=None):
        """Initialize a TargetCorrelationProcessor.

        Args:
            cluster_subspace (ClusterSubspace):
                a cluster subspace
            supercell_matrix (ndarray):
                an array representing the supercell matrix with respect to the prim
                structure.
            target_features (ndarray): optional
                target feature vector, if None given as vector of zeros is used.
            feature_weights (ndarray): optional
                Weight associated with each feature, if None given as vector of ones
        """

        if target_features is None:
            self.target = np.zeros(self.num_corr_functions)
        else:
            if len(target_features) != len(cluster_subspace):
                raise ValueError("The target feature vector must have the same length as the cluster subspace.")
            self.target = target_features

        if feature_weights is None:
            feature_weights = np.ones(len(cluster_subspace))
        elif len(feature_weights) != len(cluster_subspace):
            raise ValueError("The feature weights vector must have the same length as the cluster subspace.")

        super().__init__(cluster_subspace, supercell_matrix, feature_weights)

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
            array: change in correlation vector
        """
        corr_from_occupancy(occupancy, self.num_corr_functions, self._orbit_list)



class TargetCorrelationProcessor(Processor):
    """TargetCorrelationProcessor class to compute distance from a correlation vector
    """

    def __init__(self, cluster_subspace, supercell_matrix, target_corr):
        """Initialize a TargetCorrelationProcessor.

        Args:
            cluster_subspace (ClusterSubspace):
                a cluster subspace
            supercell_matrix (ndarray):
                an array representing the supercell matrix with respect to the
                Cluster Expansion prim structure.
            target_corr (ndarray):
                target correlation vector
        """
        super().__init__(cluster_subspace, supercell_matrix)

        self.target_corr = target_corr
