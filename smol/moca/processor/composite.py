"""Implementation of Composite processor class for a fixed size super cell.

If you have created a ClusterExpansion with additional terms, and don't want
the headache of manually and correctly spinning up the corresponding
CompositeProcessor, simply consider using the convenience class constructor
method in the ensemble classes: Ensemble.from_clusterexpansion, that will take
care of all these hickups for you.
"""

__author__ = "Luis Barroso-Luque"

import numpy as np

from smol.cofe import ClusterSubspace
from smol.moca.processor.base import Processor


class CompositeProcessor(Processor):
    """CompositeProcessor class used for mixed models.

    A Composite processor is merely a container for several different processor
    that acts as an interface such that it can be used in the same way as the
    individual processor. This can be used to mix models in any way that your
    heart desires.

    The most common use case of them all is a CompositeProcessor of a
    CEProcessor and an EwaldProcessor for use in ionic materials.

    You can add anyone of the other processor class implemented to build a
    composite processor.
    """

    def __init__(self, cluster_subspace, supercell_matrix):
        """Initialize a CompositeProcessor.

        Args:
            cluster_subspace (ClusterSubspace):
                A cluster subspace
            supercell_matrix (ndarray):
                An array representing the supercell matrix with respect to the
                Cluster Expansion prim structure.
        """
        super().__init__(cluster_subspace, supercell_matrix, None)
        self._processors = []
        self.coefs = np.empty(0)

    @property
    def processors(self):
        """Return the list of processor in composite."""
        return self._processors

    def add_processor(self, processor_class, *args, **kwargs):
        """Add a processor to composite.

        Args:
            processor_class (Processor):
                A derived class of BaseProcessor. The class, not an instance.
            *args:
                positional arguments necessary to create instance of processor
                class (excluding the subspace and supercell_matrix)
            **kwargs:
                keyword arguments necessary to create instance of processor
                class (excluding the subspace and supercell_matrix)
        """
        processor = processor_class(self._subspace, self._scmatrix,
                                    *args, **kwargs)
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
        """Compute change in property from a set of flips.

        Args:
            occupancy (ndarray):
                encoded occupancy array
            flips (list):
                list of tuples for (index of site, specie code to set)

        Returns:
            float:  property difference between inital and final states
        """
        return sum(pr.compute_property_change(occupancy, flips)
                   for pr in self._processors)

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
        features = [np.array(pr.compute_feature_vector(occupancy))
                    for pr in self._processors]
        # TODO you may be able to cut some speed by pre-allocating this
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
        updates = [np.array(pr.compute_feature_vector_change(occupancy, flips))
                   for pr in self._processors]
        # TODO you may be able to cut some speed by pre-allocating this
        return np.append(updates[0], updates[1:])

    def as_dict(self) -> dict:
        """
        Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        d = super().as_dict()
        d['processors'] = [pr.as_dict() for pr in self._processors]
        return d

    @classmethod
    def from_dict(cls, d):
        """Create a Composite from serialized MSONable dict."""
        pr = cls(ClusterSubspace.from_dict(d['cluster_subspace']),
                 np.array(d['supercell_matrix']))
        pr._processors = [Processor.from_dict(prd)
                          for prd in d['processors']]
        return pr
