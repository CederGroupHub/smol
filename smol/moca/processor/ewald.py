"""Implementation of Ewald processor class for a fixed size supercell.

An Ewald processor is optimized to compute electrostatic interaction energy
and changes in electrostatic energy from a list of local flips for use in
Monte Carlo sampling.

If you are using a Hamiltonian with a Cluster expansion and an Ewald summation
electrostatic term, you should use the CompositeProcessor with a
ClusterExpansionProcessor and an EwaldProcessor class.
"""

__author__ = "Luis Barroso-Luque"

import numpy as np
from pymatgen.analysis.ewald import EwaldSummation
from smol.cofe.space.clusterspace import ClusterSubspace
from smol.cofe.extern.ewald import EwaldTerm
from smol.moca.processor.base import Processor
from src.mc_utils import delta_ewald_single_flip


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
        ewald_summation=None,
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
                fitting coeficient to scale Ewald energy by.
            ewald_summation (EwaldSummation): optional
                pymatgen EwaldSummation instance, make sure this uses the exact
                same parameters as those used in the EwaldTerm in the
                ClusterSubspace used in the ClusterExpansion
                (i.e. same eta, real and recip cut-offs).
        """
        super().__init__(cluster_subspace, supercell_matrix, coefficient)

        self._ewald_term = ewald_term
        # Set up ewald structure and indices
        struct, inds = self._ewald_term.get_ewald_structure(self.structure)
        self._ewald_structure = struct
        self._ewald_inds = np.ascontiguousarray(inds)
        # Lazy set up Ewald Summation since it can be slow
        self._ewald = ewald_summation
        self._matrix = None  # to cache matrix for now, the use cached_prop

    @property
    def ewald_summation(self):
        """Get the pymatgen EwaldSummation object."""
        if self._ewald is None:
            self._ewald = EwaldSummation(
                self._ewald_structure,
                real_space_cut=self._ewald_term.real_space_cut,  # noqa
                recip_space_cut=self._ewald_term.recip_space_cut,  # noqa
                eta=self._ewald_term.eta,
            )
        return self._ewald

    @property  # TODO use cached_property (only for python 3.8)
    def ewald_matrix(self):
        """Get the electrostatic interaction matrix.

        The matrix used is the one set in the EwaldTerm of the given
        ClusterExpansion.
        """
        if self._matrix is None:
            matrix = self._ewald_term.get_ewald_matrix(self.ewald_summation)
            self._matrix = np.ascontiguousarray(matrix)
        return self._matrix

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
        d = super().as_dict()
        d["ewald_summation"] = self.ewald_summation.as_dict()
        d["ewald_term"] = self._ewald_term.as_dict()
        return d

    @classmethod
    def from_dict(cls, d):
        """Create an EwaldProcessor from serialized MSONable dict."""
        pr = cls(
            ClusterSubspace.from_dict(d["cluster_subspace"]),
            np.array(d["supercell_matrix"]),
            ewald_term=EwaldTerm.from_dict(d["ewald_term"]),
            coefficient=d["coefficients"],
            ewald_summation=EwaldSummation.from_dict(d["ewald_summation"]),
        )
        return pr
