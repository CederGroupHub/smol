"""Implementation of Composite processor class for a fixed size supercell.

If you have created a ClusterExpansion with additional terms, and don't want
the headache of manually and correctly setting up the corresponding
CompositeProcessor, simply consider using the convenience class constructor
method in the ensemble classes: Ensemble.from_clusterexpansion, that will take
care of all these hiccups for you.
"""

__author__ = "Luis Barroso-Luque"

import numpy as np

from smol.cofe.space.clusterspace import ClusterSubspace
from smol.moca.processor.base import Processor


class CompositeProcessor(Processor):
    """CompositeProcessor class used for mixed models.

    A Composite processor is merely a container for many different processors
    that acts as an interface such that it can be used in the same way as an
    individual processor. This can be used to mix models in any way that your
    heart desires.

    The most common use case is a CompositeProcessor of a
    ClusterExpansionProcessor and an EwaldProcessor for use in ionic materials
    when the cluster expansion contains an Ewald term.

    You can add any of the other processor class implemented to build a
    composite processor.

    It is recommended to use the :code:`from_cluster_expansion` to create an
    ensemble and the underlying processor automatically created rather than
    directly creating a processor. This will take care of creating the correct
    :class:`CompositeProcessor` or :class:`ClusterExpansionProcessor` for you.
    """

    def __init__(self, cluster_subspace, supercell_matrix):
        """Initialize a CompositeProcessor.

        Args:
            cluster_subspace (ClusterSubspace):
                a cluster subspace
            supercell_matrix (ndarray):
                an array representing the supercell matrix with respect to the
                cluster expansion prim structure.
        """
        super().__init__(cluster_subspace, supercell_matrix, None)
        self._processors = []
        self.coefs = np.empty(0)

    @property
    def processors(self):
        """Return the list of processors in CompositeProcessor."""
        return self._processors

    def add_processor(self, processor):
        """Add a processor to composite.

        Args:
            processor (Processor):
                processor to add. All processors must have the same
                ClusterSubspace and supercell matrix.
        """
        if isinstance(processor, CompositeProcessor):
            raise AttributeError(
                "A CompositeProcessor can not be added into "
                "another CompositeProcessor"
            )
        if self.cluster_subspace != processor.cluster_subspace:
            raise ValueError(
                "The cluster subspace of the processor to be"
                " added does not match the one of this "
                "CompositeProcessor."
            )
        if not np.array_equal(self._scmatrix, processor.supercell_matrix):
            raise ValueError(
                "The supercell matrix of the processor to be"
                " added does not match the one of this "
                "CompositeProcessor."
            )
        self._processors.append(processor)
        self.coefs = np.append(self.coefs, processor.coefs)

    def compute_property(self, occupancy):
        """Compute the value of the property for the given occupancy array.

        Args:
            occupancy (ndarray):
                encoded occupancy array
        Returns:
            float: predicted property
        """
        return sum(pr.compute_property(occupancy) for pr in self._processors)

    def compute_property_change(self, occupancy, flips):
        """Compute the change in property from a set of flips.

        Args:
            occupancy (ndarray):
                encoded occupancy array
            flips (list):
                list of tuples for (index of site, specie code to set)

        Returns:
            float:  property difference between initial and final states
        """
        return sum(
            pr.compute_property_change(occupancy, flips) for pr in self._processors
        )

    def compute_feature_vector(self, occupancy):
        """Compute the feature vector for a given occupancy array.

        Each entry in the correlation vector corresponds to a particular
        symmetrically distinct bit ordering.

        Args:
            occupancy (ndarray):
                encoded occupation array

        Returns:
            array: correlation vector
        """
        features = [
            np.array(pr.compute_feature_vector(occupancy)) for pr in self._processors
        ]
        return np.append(features[0], features[1:])

    def compute_feature_vector_change(self, occupancy, flips):
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
        updates = [
            np.array(pr.compute_feature_vector_change(occupancy, flips))
            for pr in self._processors
        ]
        # TODO you may be able to cut some speed by pre-allocating this
        return np.append(updates[0], updates[1:])

    def as_dict(self) -> dict:
        """
        Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        proc_d = super().as_dict()
        proc_d["processors"] = [pr.as_dict() for pr in self._processors]
        return proc_d

    @classmethod
    def from_dict(cls, d):
        """Create a CompositeProcessor from serialized MSONable dict."""
        # pylint: disable=duplicate-code
        proc = cls(
            ClusterSubspace.from_dict(d["cluster_subspace"]),
            np.array(d["supercell_matrix"]),
        )
        for prd in d["processors"]:
            proc.add_processor(Processor.from_dict(prd))
        return proc
