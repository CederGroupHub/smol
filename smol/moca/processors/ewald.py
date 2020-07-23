import numpy as np
from pymatgen.analysis.ewald import EwaldSummation
from src.mc_utils import corr_from_occupancy, delta_ewald_single_flip

from smol.cofe import ClusterExpansion
from smol.cofe.extern import EwaldTerm
from smol.moca import CEProcessor
from smol.moca.processors.base import BaseProcessor


class EwaldCEProcessor(CEProcessor, BaseProcessor):  # is this troublesome?
    """Processor for CE's including an EwaldTerm.

    A subclass of the CEProcessor class that handles changes for the
    electrostatic interaction energy in an additional Ewald Summation term.
    Make sure that the ClusterExpansion object used has an EwaldTerm,
    otherwise this will not work.
    """

    def __init__(self, cluster_expansion, supercell_matrix,
                 optimize_indicator=False, ewald_summation=None):
        """Initialize an EwaldCEProcessor.

        Args:
            cluster_expansion (ClusterExpansion):
                A fitted cluster expansion representing a Hamiltonian
            supercell_matrix (ndarray):
                An array representing the supercell matrix with respect to the
                Cluster Expansion prim structure.
            optimize_indicator (bool):
                When using an indicator basis, set the delta_corr function to
                the indicator optimize function. This can make MC steps faster.
            ewald_summation (EwaldSummation): optional
                pymatgen EwaldSummation instance, make sure this uses the exact
                same parameters as those used in the EwaldTerm in the cluster
                Expansion (i.e. same eta, real and recip cuts).
        """
        super().__init__(cluster_expansion, supercell_matrix,
                         optimize_indicator)

        n_terms = len(cluster_expansion.cluster_subspace.external_terms)
        if n_terms != 1:
            raise AttributeError('The provided ClusterExpansion must have only'
                                 ' one external term (an EwaldTerm). The one '
                                 f'provided has {n_terms} external terms.')
        self._ewald_term = cluster_expansion.cluster_subspace.external_terms[0]
        if not isinstance(self._ewald_term, EwaldTerm):
            raise TypeError('The external term in the provided '
                            'ClusterExpansion must be an EwaldTerm.'
                            f'Got {type(self._ewald_term)}')

        # Set up ewald structure and indices
        struct, inds = self._ewald_term.get_ewald_structure(self.structure)
        self._ewald_structure = struct
        self._ewald_inds = np.ascontiguousarray(inds)
        # Lazy set up Ewald Summation since it can be slow
        self.__ewald = ewald_summation
        self.__matrix = None  # to cache matrix for now, the use cached_prop

    @property
    def ewald_summation(self):
        """Get the pymatgen EwaldSummation object."""
        if self.__ewald is None:
            self.__ewald = EwaldSummation(self._ewald_structure,
                                          real_space_cut=self._ewald_term.real_space_cut,  # noqa
                                          recip_space_cut=self._ewald_term.recip_space_cut,  # noqa
                                          eta=self._ewald_term.eta)
        return self.__ewald

    @property  # TODO use cached_property (only for python 3.8)
    def ewald_matrix(self):
        """Get the electrostatic interaction matrix.

        The matrix used is the one set in the EwaldTerm of the given
        ClusterExpansion.
        """
        if self.__matrix is None:
            matrix = self._ewald_term.get_ewald_matrix(self.ewald_summation)
            self.__matrix = np.ascontiguousarray(matrix)
        return self.__matrix

    def compute_correlation(self, occu):
        """Compute the correlation vector for a given occupancy array.

        Each entry in the correlation vector corresponds to a particular
        symmetrically distinct bit ordering. The last value of the correlation
        is the ewald summation value.

        Args:
            occu (ndarray):
                encoded occupation array

        Returns:
            ndarray: correlation vector
        """
        ce_corr = corr_from_occupancy(occu, self.n_orbit_functions,
                                      self._orbit_list)
        ewald_val = self._ewald_value_from_occupancy(occu)/self.size
        return np.append(ce_corr, ewald_val)

    def _ewald_value_from_occupancy(self, occu):
        """Compute the electrostatic interaction energy."""
        matrix = self.ewald_matrix
        ew_occu = self._ewald_term.get_ewald_occu(occu,
                                                  matrix.shape[0],
                                                  self._ewald_inds)
        interactions = np.sum(matrix[ew_occu, :][:, ew_occu])
        return interactions

    def _delta_corr(self, flips, occu):
        """Compute the change in the correlation vector from list of flips.

        Args:
            flips list(tuple):
                 list of tuples with two elements. Each tuple represents a
                 single flip where the first element is the index of the site
                 in the occupancy array and the second element is the index
                 for the new species to place at that site.
            occu (ndarray):
                encoded occupancy array

        Returns:
            array
        """
        occu_i = occu
        delta_corr = np.zeros(self.n_orbit_functions + 1)
        for f in flips:
            occu_f = occu_i.copy()
            occu_f[f[0]] = f[1]
            orbits = self._orbits_by_sites[f[0]]
            delta_corr[:-1] += self._dcorr_single_flip(occu_f, occu_i,
                                                       self.n_orbit_functions,
                                                       orbits)
            delta_corr[-1] += delta_ewald_single_flip(occu_f, occu_i,
                                                      self.ewald_matrix,
                                                      self._ewald_inds,
                                                      f[0],
                                                      self.size)
            occu_i = occu_f

        return delta_corr

    def as_dict(self) -> dict:
        """
        Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        d = super().as_dict()
        d['_ewald'] = self.ewald_summation.as_dict()
        return d

    @classmethod
    def from_dict(cls, d):
        """Create a CEProcessor from serialized MSONable dict."""
        pr = cls(ClusterExpansion.from_dict(d['cluster_expansion']),
                 np.array(d['supercell_matrix']),
                 optimize_indicator=d['indicator'],
                 ewald_summation=EwaldSummation.from_dict(d['_ewald']))
        return pr