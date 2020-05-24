"""Implementation of processor classes for a fixed size super cell.

A processor is optimized to compute correlation vectors and local changes in
correlation vectors. This class allows the use a cluster expansion hamiltonian
to run Monte Carlo based simulations.

If you are using a Hamiltonian with an Ewald summation electrostatic term, you
should use the EwaldCEProcessor class to handle changes in the electrostatic
interaction energy.
"""

__author__ = "Luis Barroso-Luque"

import numpy as np
from collections import defaultdict, OrderedDict
from monty.json import MSONable
from pymatgen import Structure, PeriodicSite
from pymatgen.analysis.ewald import EwaldSummation
from smol.cofe import ClusterExpansion
from smol.cofe.extern import EwaldTerm
from smol.cofe.configspace.utils import get_bits, get_bits_w_concentration
from src.ce_utils import (corr_from_occupancy, general_delta_corr_single_flip,
                          delta_ewald_single_flip,
                          indicator_delta_corr_single_flip)


class CEProcessor(MSONable):
    """
    A processor allows an ensemble class to generate a Markov chain
    for sampling thermodynamic properties from a cluster expansion
    Hamiltonian.
    Think of this as fixed size supercell optimized to calculate correlation
    vectors and local changes to correlation vectors from site flips.
    """

    def __init__(self, cluster_expansion, supercell_matrix,
                 optimize_indicator=False):
        """
        Args:
            cluster_expansion (ClusterExpansion):
                A fitted cluster expansion representing a Hamiltonian
            supercell_matrix (array):
                An array representing the supercell matrix with respect to the
                Cluster Expansion prim structure.
            optimize_indicator (bool):
                When using an indicator basis, set the delta_corr function to
                the indicator optimize function. This can make MC steps faster.
                Make sure your cluster expansion was indeed fit with an
                indicator basis set, otherwise your MC results are no good.
        """

        # set the dcorr_single_flip function
        self.indicator_opt = optimize_indicator
        self.dcorr_single_flip = indicator_delta_corr_single_flip \
            if optimize_indicator \
            else general_delta_corr_single_flip

        # the only reason to keep the CE is for the MSONable from_dict
        self.cluster_expansion = cluster_expansion
        self._subspace = cluster_expansion.cluster_subspace
        self._ecis = cluster_expansion.ecis
        self.structure = self._subspace.structure.copy()
        self.structure.make_supercell(supercell_matrix)
        self.supercell_matrix = supercell_matrix

        # this can be used (maybe should) to check if a flip is valid
        exp_bits = get_bits_w_concentration(cluster_expansion.expansion_structure)  # noqa
        self.unique_bits = tuple(OrderedDict(bits) for bits in
                                 set(tuple(bits.items()) for bits in exp_bits))

        self.bits = get_bits(self.structure)
        self.size = self._subspace.num_prims_from_matrix(supercell_matrix)
        self.n_orbit_functions = self._subspace.n_bit_orderings
        self._orbit_inds = self._subspace.supercell_orbit_mappings(supercell_matrix)  # noqa

        # List of orbit information and supercell site indices to compute corr
        self._orbit_list = []
        # Dictionary of orbits by site index and information
        # necessary to compute local changes in correlation vectors from flips
        self._orbits_by_sites = defaultdict(list)
        # Store the orbits grouped by site index in the structure,
        # to be used by delta_corr. We also store a reduced index array,
        # where only the rows with the site index are stored. The ratio is
        # needed because the correlations are averages over the full inds
        # array.
        for orbit, inds in self._orbit_inds:
            self._orbit_list.append((orbit.bit_id, orbit.bit_combos,
                                     orbit.bases_array, inds))
            for site_ind in np.unique(inds):
                in_inds = np.any(inds == site_ind, axis=-1)
                ratio = len(inds) / np.sum(in_inds)
                self._orbits_by_sites[site_ind].append((orbit.bit_id, ratio,
                                                        orbit.bit_combos,
                                                        orbit.bases_array,
                                                        inds[in_inds]))

    def compute_property(self, occu):
        """
        Computes the total value of the property corresponding to the CE
        for the given occupancy array

        Args:
            occu (array):
                encoded occupancy array
        Returns: predicted property
            float
        """
        return np.dot(self.compute_correlation(occu), self._ecis) * self.size

    def compute_property_change(self, occu, flips):
        """
        Compute change in property from a set of flips.

        Args:
            occu (array):
                encoded occupancy array
            flips (list):
                list of tuples for (index of site, specie code to set)

        Returns:
            float
        """
        return np.dot(self.delta_corr(flips, occu), self._ecis) * self.size

    def compute_correlation(self, occu):
        """
        Computes the correlation vector for a given occupancy array.
        Each entry in the correlation vector corresponds to a particular
        symmetrically distinct bit ordering.

        Args:
            occu (array):
                encoded occupation array

        Returns: Correlation vector
            array
        """
        return corr_from_occupancy(occu, self.n_orbit_functions,
                                   self._orbit_list)

    def occupancy_from_structure(self, structure):
        """
        Gets the occupancy array for a given structure. The structure must
        strictly be a supercell of the prim according to the processor's
        supercell matrix

        Args:
            structure (Structure):
                A pymatgen structure (related to the cluster-expansion prim
                by the supercell matrix passed to the processor)
        Returns: encoded occupancy array
            list
        """
        occu = self._subspace.occupancy_from_structure(structure,
                                                       scmatrix=self.supercell_matrix)  # noqa
        return self.encode_occupancy(occu)

    def structure_from_occupancy(self, occu):
        """
        Get pymatgen.Structure from an occupancy array.

        Args:
            occu (array):
                encoded occupancy array

        Returns:
            Structure
        """
        occu = self.decode_occupancy(occu)
        sites = []
        for sp, s in zip(occu, self.structure):
            if sp != 'Vacancy':
                site = PeriodicSite(sp, s.frac_coords, self.structure.lattice)
                sites.append(site)
        return Structure.from_sites(sites)

    def encode_occupancy(self, occu):
        """
        Encode occupancy array of species str to ints.
        """
        ec_occu = np.array([bit.index(sp) for bit, sp in zip(self.bits, occu)])
        return ec_occu

    def decode_occupancy(self, enc_occu):
        """
        Decode encoded occupancy array of int to species str.
        """
        occu = [bit[i] for i, bit in zip(enc_occu, self.bits)]
        return occu

    def delta_corr(self, flips, occu):
        """
        Computes the change in the correlation vector from a list of site
        flips.

        Args:
            flips list(tuple):
                list of tuples with two elements. Each tuple represents a
                single flip where the first element is the index of the site
                in the occupancy array and the second element is the index
                for the new species to place at that site.
            occu (array):
                encoded occupancy array

        Returns:
            array
        """

        occu_i = occu.copy()
        delta_corr = np.zeros(self.n_orbit_functions)
        for f in flips:
            occu_f = occu_i.copy()
            occu_f[f[0]] = f[1]
            orbits = self._orbits_by_sites[f[0]]
            delta_corr += self.dcorr_single_flip(occu_f, occu_i,
                                                 self.n_orbit_functions,
                                                 orbits)
            occu_i = occu_f

        return delta_corr

    @classmethod
    def from_dict(cls, d):
        """
        Creates CEProcessor from serialized MSONable dict
        """
        return cls(ClusterExpansion.from_dict(d['cluster_expansion']),
                   np.array(d['supercell_matrix']),
                   optimize_indicator=d['indicator'])

    def as_dict(self) -> dict:
        """
        Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        d = {'@module': self.__class__.__module__,
             '@class': self.__class__.__name__,
             'cluster_expansion': self.cluster_expansion.as_dict(),
             'supercell_matrix': self.supercell_matrix.tolist(),
             'indicator': self.indicator_opt}
        return d


class EwaldCEProcessor(CEProcessor):
    """
    A subclass of the CEProcessor class that handles changes for the
    electrostatic interaction energy in an additional Ewald Summation term.
    Make sure that the ClusterExpansion object used has an EwaldTerm,
    otherwise this will not work.
    """

    def __init__(self, cluster_expansion, supercell_matrix,
                 optimize_indicator=False):
        """
        Args:
            cluster_expansion (ClusterExpansion):
                A fitted cluster expansion representing a Hamiltonian
            supercell_matrix (array):
                An array representing the supercell matrix with respect to the
                Cluster Expansion prim structure.
            optimize_indicator (bool):
                When using an indicator basis, set the delta_corr function to
                the indicator optimize function. This can make MC steps faster.
        """

        super().__init__(cluster_expansion, supercell_matrix)

        n_terms = len(cluster_expansion.cluster_subspace.external_terms)
        if n_terms != 1:
            raise AttributeError('The provided ClusterExpansion must have only'
                                 ' one external term (an EwaldTerm). The one '
                                 f'provided has {n_terms} external terms.')
        self._ewald_term = cluster_expansion.cluster_subspace.external_terms[0]
        if not isinstance(self._ewald_term, EwaldTerm):
            raise TypeError('The external term in the provided '
                            'ClusterExpansion must be an EwaldTerm.')

        # Set up Ewald Summation
        struct, inds = self._ewald_term._get_ewald_structure(self.structure)
        self._ewald_structure, self._ewald_inds = struct, inds
        self._ewald = EwaldSummation(struct, self._ewald_term.eta)
        # This line extending the ewald matrix by one dimension is done only
        # to use the original cython delta_ewald function that took additional
        # matrices for partial_ems when using the legacy inv_r=True option
        # This hassle can be removed if the delta_ewald cython function is
        # updated accordingly
        self._all_ewalds = np.array([self._ewald.total_energy_matrix])

    def compute_correlation(self, occu):
        """
        Computes the correlation vector for a given occupancy array.
        Each entry in the correlation vector corresponds to a particular
        symmetrically distinct bit ordering. The last value of the correlation
        is the ewald summation value.

        Args:
            occu (array):
                encoded occupation vector

        Returns: Correlation vector
            array
        """

        ce_corr = corr_from_occupancy(occu, self.n_orbit_functions,
                                      self._orbit_list)
        ewald_corr = self._ewald_correlation(occu)
        return np.append(ce_corr, ewald_corr)

    def delta_corr(self, flips, occu):
        """
        Computes the change in the correlation vector from a list of site
        flips

        Args:
            flips list(tuple):
                 list of tuples with two elements. Each tuple represents a
                 single flip where the first element is the index of the site
                 in the occupancy array and the second element is the index
                 for the new species to place at that site.
            occu (array):
                encoded occupancy array

        Returns:
            array
        """
        occu_i = occu.copy()
        delta_corr = np.zeros(self.n_orbit_functions + 1)
        for f in flips:
            occu_f = occu_i.copy()
            occu_f[f[0]] = f[1]
            orbits = self._orbits_by_sites[f[0]]
            delta_corr[:-1] += self.dcorr_single_flip(occu_f, occu_i,
                                                      self.n_orbit_functions,
                                                      orbits)
            delta_corr[-1] += delta_ewald_single_flip(occu_f, occu_i,
                                                      self.n_orbit_functions,
                                                      orbits,
                                                      f[0], f[1],
                                                      self._all_ewalds,
                                                      self._ewald_inds,
                                                      self.size)
            occu_i = occu_f

        return delta_corr

    def _ewald_correlation(self, occu):
        """
        Computes the normalized by prim electrostatic interaction energy
        """
        num_ewald_sites = len(self._ewald_structure)
        matrix = self._ewald.total_energy_matrix
        ew_occu = self._ewald_term._get_ewald_occu(occu, num_ewald_sites,
                                                   self._ewald_inds)
        interactions = np.sum(matrix[ew_occu, :][:, ew_occu])
        return interactions/self.size
