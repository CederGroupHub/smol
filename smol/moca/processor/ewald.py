"""Implementation of Ewald processor class for a fixed size supercell.

An Ewald processor is optimized to compute electrostatic interaction energy
and changes in electrostatic energy from a list of local flips for use in
Monte Carlo sampling.

If you are using a Hamiltonian with a Cluster expansion and an Ewald summation
electrostatic term, you should use the CompositeProcessor with a
ClusterExpansionProcessor and an EwaldProcessor class.
"""

__author__ = "Luis Barroso-Luque"

import warnings
from functools import cached_property, lru_cache

import numpy as np
from pymatgen.analysis.ewald import EwaldSummation

from smol.cofe.extern.ewald import EwaldTerm
from smol.cofe.space.clusterspace import ClusterSubspace
from smol.moca.processor.base import Processor
from smol.utils.cluster.ewald import delta_ewald_single_flip


class EwaldProcessor(Processor):
    """Processor for CE's including an EwaldTerm.

    A Processor class that handles changes for the electrostatic interaction
    energy using an Ewald Summation term.
    """

    def __init__(
        self,
        cluster_subspace,
        supercell_matrix,
        ewald_term,
        coefficient=1.0,
    ):
        """Initialize an EwaldProcessor.

        Args:
            cluster_subspace (ClusterSubspace):
                a cluster subspace.
            supercell_matrix (ndarray):
                an array representing the supercell matrix with respect to the
                Cluster Expansion prim structure.
            ewald_term (EwaldTerm):
                an instance of EwaldTerm to compute electrostatic energies.
            coefficient (float):
                Fitting coefficient to scale Ewald energy by.
        """
        contains_ewald = False
        for term in cluster_subspace.external_terms:
            if isinstance(term, EwaldTerm):
                contains_ewald = True
                break

        if not contains_ewald:
            cluster_subspace.add_external_term(EwaldTerm())
            warnings.warn(
                message="Warning: cluster subspace does not contain "
                "an Ewald term. Creating a default Ewald "
                "term and adding to cluster subspace"
            )
        super().__init__(cluster_subspace, supercell_matrix, coefficient)

        self._ewald_term = ewald_term
        # Set up ewald structure and indices
        struct, inds = self._ewald_term.get_ewald_structure(self.structure)
        self._ewald_structure = struct
        self._ewald_inds = np.ascontiguousarray(inds)

    @cached_property
    def ewald_summation(self):
        """Get the pymatgen EwaldSummation object."""
        ewald_summation = EwaldSummation(
            self._ewald_structure,
            real_space_cut=self._ewald_term.real_space_cut,
            recip_space_cut=self._ewald_term.recip_space_cut,
            eta=self._ewald_term.eta,
        )
        return ewald_summation

    @property
    @lru_cache(maxsize=None)
    def ewald_matrix(self):
        """Get the electrostatic interaction matrix.

        The matrix used is the one set in the EwaldTerm of the given
        ClusterExpansion.
        """
        matrix = self._ewald_term.get_ewald_matrix(self.ewald_summation)
        matrix = np.ascontiguousarray(matrix)
        return matrix

    def compute_property(self, occupancy):
        """Compute the Ewald electrostatic energy for a given occupancy array.

        Args:
            occupancy (ndarray):
                encoded occupancy array
        Returns:
            float: Ewald electrostatic energy
        """
        return self.coefs * self.compute_feature_vector(occupancy)

    def compute_property_change(self, occupancy, flips):
        """Compute change in electrostatic energy from a set of flips.

        Args:
            occupancy (ndarray):
                encoded occupancy array
            flips (list):
                list of tuples for (index of site, specie code to set)

        Returns:
            float: electrostatic energy change
        """
        return self.coefs * self.compute_feature_vector_change(occupancy, flips)

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
        ew_occu = self._ewald_term.get_ewald_occu(
            occupancy, self.ewald_matrix.shape[0], self._ewald_inds
        )
        return np.sum(self.ewald_matrix[ew_occu, :][:, ew_occu])

    def compute_feature_vector_change(self, occupancy, flips):
        """
        Compute the change in the feature vector from a list of flips.

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
        delta_energy = 0
        for f in flips:
            occu_f = occu_i.copy()
            occu_f[f[0]] = f[1]
            delta_energy += delta_ewald_single_flip(
                occu_f, occu_i, self.ewald_matrix, self._ewald_inds, f[0]
            )
            occu_i = occu_f
        return delta_energy

    def as_dict(self) -> dict:
        """
        Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        ewald_d = super().as_dict()
        ewald_d["ewald_summation"] = self.ewald_summation.as_dict()
        ewald_d["ewald_term"] = self._ewald_term.as_dict()
        return ewald_d

    @classmethod
    def from_dict(cls, d):
        """Create a EwaldProcessor from serialized MSONable dict."""
        # pylint: disable=duplicate-code
        proc = cls(
            ClusterSubspace.from_dict(d["cluster_subspace"]),
            np.array(d["supercell_matrix"]),
            ewald_term=EwaldTerm.from_dict(d["ewald_term"]),
            coefficient=d["coefficients"],
        )
        proc.ewald_summation = EwaldSummation.from_dict(d["ewald_summation"])

        return proc
