"""Implementation of target feature processors.

Feature metric processors are used to compute a distance between the feature values
of a particular configuration and a target feature vector. This functionality is useful
to obtain special structures, such as SQS, SSOS, etc.
"""

__author__ = "Luis Barroso-Luque"


import numpy as np

from smol.cofe.space.clusterspace import ClusterSubspace
from smol.moca.processor.base import Processor


def create_feature_metric_processor(processor_class):
    """Create a feature metric processor class derived from a given processor class.

    The FeatureMetricProcessor is derived from the given processor_class, meaning
    that the features to be computed are those associated with that processor.

    Args:
        processor_class (class):
            processor class to use.
    Returns:
        FeatureMetricProcessor:
            A FeatureMetricPocessor class derived from the given processor class.
    """
    if not issubclass(processor_class, Processor):
        raise TypeError("processor_class must be a Processor class")

    class FeatureMetricProcessor(processor_class):
        """FeatureMetricProcessor class to compute distance from a feature vector.

        The metric used to measure distance is,
        d = -wL + ||f - f_T||_1
        As proposed in the following publication,
        https://doi.org/10.1016/j.calphad.2013.06.006
        """

        def __init__(
            self,
            cluster_subspace,
            supercell_matrix,
            target_features=None,
            match_weight=None,
            **processor_kwargs
        ):
            """Initialize a TargetCorrelationProcessor.

            Args:
                cluster_subspace (ClusterSubspace):
                    a cluster subspace
                supercell_matrix (ndarray):
                    an array representing the supercell matrix with respect to the
                    Cluster Expansion prim structure.
                target_features (ndarray): optional
                    target feature vector, if None given as vector of zeros is used.
                match_weight (float): optional
                    weight for the in the wL term above. That is how much to weight the
                    largest diameter below which all features are matched exactly.
                    Set to any number >0 to use this term. If None, then the term is
                    ignored.
                processor_kwargs (dict): optional
                    additional keyword arguments to instantiate the underlying processor
                    being inherited from.
            """
            super().__init__(cluster_subspace, supercell_matrix, **processor_kwargs)

            if target_features is None:
                self.target = np.zeros(len(self.coefs))
            else:
                if len(target_features) != len(self.coefs):
                    raise ValueError(
                        "The target feature vector must have the same length as those computed by the processor."
                    )
                self.target = target_features

            if match_weight is not None:
                if match_weight <= 0:
                    raise ValueError("The match weight must be a positive number.")
                self.match_weight = match_weight

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
            return np.abs(super().compute_feature_vector(occupancy) - self.target)

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

    return FeatureMetricProcessor
