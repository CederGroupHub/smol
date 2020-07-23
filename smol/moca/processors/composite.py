"""Implementation of Composite processor class for a fixed size super cell.

A Composite processor is merely a container for several different processors
that acts as an interface such that it can be used in the same way as the
individual processor. This can be used to mix models in any way that your
heart desires.

The most common use case of them all is a CompositeProcessor of a CEProcessor
and an EwaldProcessor for use in ionic materials.

If you have created a ClusterExpansion with additional terms, and don't want
the headache of manually and correctly spinning up the corresponding
CompositeProcessor, simply consider using the convenience class constructor
method in the ensemble classes: Ensemble.from_clusterexpansion, that will take
care of all these hickups for you.
"""

__author__ = "Luis Barroso-Luque"

import numpy as np

from smol.cofe import ClusterSubspace
from smol.moca.processors.base import BaseProcessor


class CompositeProcessor(BaseProcessor):
    """CompositeProcessor class used for mixed models.

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
        super().__init__(cluster_subspace, supercell_matrix)
        self._processors = []

    @property
    def processors(self):
        """Return the list of processors in composite."""
        return [pr for _, pr in self._processors]

    @property
    def mixing_coefficients(self):
        """Return a list of the mixing coefficients for each processor."""
        return [coef for coef, _ in self._processors]

    def add_processor(self, processor_class, mixing_coefficient=1,
                      *args, **kwargs):
        """Add a processor to composite.

        Args:
            processor_class (BaseProcessor):
                A derived class of BaseProcessor. The class, not an instance.
            mixing_coefficient (float)
                Coefficient to multiply the values of computed properties when
                calculating the total for the composite.
            *args:
                positional arguments necessary to create instance of processor
                class (excluding the subspace and supercell_matrix)
            **kwargs:
                keyword arguments necessary to create instance of processor
                class (excluding the subspace and supercell_matrix)
        """
        processor = processor_class(self._subspace, self._scmatrix,
                                    *args, **kwargs)
        self._processors.append((mixing_coefficient, processor))

    def compute_property(self, occupancy):
        """Compute the value of the property for the given occupancy array.

        Args:
            occupancy (ndarray):
                encoded occupancy array
        Returns:
            float: predicted property
        """
        return sum([mix * pr.compute_property(occupancy)
                    for mix, pr in self._processors])

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
        return sum([mix * pr.compute_property_change(occupancy, flips)
                    for mix, pr in self._processors])

    def as_dict(self) -> dict:
        """
        Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        d = super().as_dict()
        d['_processors'] = [(mix, pr.as_dict())
                            for mix, pr in self._processors]
        return d

    @classmethod
    def from_dict(cls, d):
        """Create a Composite from serialized MSONable dict."""
        pr = cls(ClusterSubspace.from_dict(d['cluster_subspace']),
                 np.array(d['supercell_matrix']))
        pr._processors = [(mix, BaseProcessor.from_dict(prd))
                          for mix, prd in d['_processors']]
