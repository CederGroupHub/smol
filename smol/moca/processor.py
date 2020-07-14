"""Implementation of processor classes for a fixed size super cell.

A processor is optimized to compute correlation vectors and local changes in
correlation vectors. This class allows the use a cluster expansion hamiltonian
to run Monte Carlo based simulations.

If you are using a Hamiltonian with an Ewald summation electrostatic term, you
should use the EwaldCEProcessor class to handle changes in the electrostatic
interaction energy.
"""

__author__ = "Luis Barroso-Luque"

from abc import ABCMeta, abstractmethod
import numpy as np
from collections import defaultdict, OrderedDict
from monty.json import MSONable
from pymatgen import Structure, PeriodicSite
from pymatgen.analysis.ewald import EwaldSummation
from smol.cofe import ClusterExpansion
from smol.cofe.extern import EwaldTerm
from smol.cofe.configspace.utils import get_site_spaces
from src.mc_utils import (corr_from_occupancy, general_delta_corr_single_flip,
                          delta_ewald_single_flip,
                          indicator_delta_corr_single_flip)


class BaseProcessor(MSONable, metaclass=ABCMeta):
    """Abstract base class for processors.

    A processor is used to provide a quick way to calculated energy differences
    (probability ratio's) between two adjacent configurational states.
    """

    @abstractmethod
    def compute_property_change(self, occu, flips):
        """Compute change in property from a set of flips.

        Args:
            occu (ndarray):
                encoded occupancy array
            flips (list):
                list of tuples for (index of site, specie code to set)

        Returns:
            float:  property difference between inital and final states
        """
        return

    def as_dict(self) -> dict:
        """
        Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        d = {'@module': self.__class__.__module__,
             '@class': self.__class__.__name__}
        return d

    @classmethod
    def from_dict(cls, d):
        """Create a CEProcessor from serialized MSONable dict."""
        # is this good design?
        try:
            for derived in cls.__subclasses__():
                if derived.__name__ == d['@class']:
                    return derived.from_dict(d)
        except KeyError:
            raise NameError(f"Unable to instantiate {d['@class']}.")


class CEProcessor(BaseProcessor):
    """CEProcessor class to use a ClusterExpansion in MC simulations.

    A processor allows an ensemble class to generate a Markov chain
    for sampling thermodynamic properties from a cluster expansion
    Hamiltonian.

    Think of this as fixed size supercell optimized to calculate correlation
    vectors and local changes to correlation vectors from site flips.
    """

    def __init__(self, cluster_expansion, supercell_matrix,
                 optimize_indicator=False):
        """Initialize a CEProcessor.

        Args:
            cluster_expansion (ClusterExpansion):
                A fitted cluster expansion representing a Hamiltonian
            supercell_matrix (ndarray):
                An array representing the supercell matrix with respect to the
                Cluster Expansion prim structure.
            optimize_indicator (bool):
                When using an indicator basis, set the delta_corr function to
                the indicator optimize function. This can make MC steps faster.
                Make sure your cluster expansion was indeed fit with an
                indicator basis set, otherwise your MC results are no good.
        """
        # the only reason to keep the CE is for the MSONable from_dict
        self.cluster_expansion = cluster_expansion
        self.coefs = cluster_expansion.coefs
        self.subspace = cluster_expansion.cluster_subspace
        self.structure = self.subspace.structure.copy()
        self.structure.make_supercell(supercell_matrix)
        self.supercell_matrix = supercell_matrix

        # this can be used (maybe should) to check if a flip is valid
        site_spaces = get_site_spaces(cluster_expansion.expansion_structure,
                                      include_measure=True)
        self.unique_site_spaces = tuple(OrderedDict(space) for space in
                                        set(tuple(spaces.items())
                                            for spaces in site_spaces))

        self.allowed_species = get_site_spaces(self.structure)
        self.size = self.subspace.num_prims_from_matrix(supercell_matrix)
        self.n_orbit_functions = self.subspace.n_bit_orderings

        # set the dcorr_single_flip function
        self.indicator_opt = optimize_indicator
        self._dcorr_single_flip = indicator_delta_corr_single_flip \
            if optimize_indicator \
            else general_delta_corr_single_flip

        self._orbit_inds = self.subspace.supercell_orbit_mappings(supercell_matrix)  # noqa
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
        """Compute the value of the property for the given occupancy array.

        The property fitted to the corresponding to the CE.

        Args:
            occu (ndarray):
                encoded occupancy array
        Returns:
            float: predicted property
        """
        return np.dot(self.compute_correlation(occu), self.coefs) * self.size

    def compute_property_change(self, occu, flips):
        """Compute change in property from a set of flips.

        Args:
            occu (ndarray):
                encoded occupancy array
            flips (list):
                list of tuples for (index of site, specie code to set)

        Returns:
            float:  property difference between inital and final states
        """
        return np.dot(self._delta_corr(flips, occu), self.coefs) * self.size

    def compute_correlation(self, occu):
        """Compute the correlation vector for a given occupancy array.

        Each entry in the correlation vector corresponds to a particular
        symmetrically distinct bit ordering.

        Args:
            occu (ndarray):
                encoded occupation array

        Returns:
            array: correlation vector
        """
        return corr_from_occupancy(occu, self.n_orbit_functions,
                                   self._orbit_list)

    def occupancy_from_structure(self, structure):
        """Get the occupancy array for a given structure.

        The structure must strictly be a supercell of the prim according to the
        processor's supercell matrix.

        Args:
            structure (Structure):
                A pymatgen structure (related to the cluster-expansion prim
                by the supercell matrix passed to the processor)
        Returns: encoded occupancy array
            list
        """
        occu = self.subspace.occupancy_from_structure(structure,
                                                      scmatrix=self.supercell_matrix)  # noqa
        return self.encode_occupancy(occu)

    def structure_from_occupancy(self, occu):
        """Get pymatgen.Structure from an occupancy array.

        Args:
            occu (ndarray):
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
        """Encode occupancy array of species str to ints."""
        return np.array([species.index(sp) for species, sp
                        in zip(self.allowed_species, occu)])

    def decode_occupancy(self, enc_occu):
        """Decode an encoded occupancy array of int to species str."""
        return [species[i] for i, species in
                zip(enc_occu, self.allowed_species)]

    def _delta_corr(self, flips, occu):
        """
        Compute the change in the correlation vector from a list of flips.

        Args:
            flips list(tuple):
                list of tuples with two elements. Each tuple represents a
                single flip where the first element is the index of the site
                in the occupancy array and the second element is the index
                for the new species to place at that site.
            occu (ndarray):
                encoded occupancy array

        Returns:
            array: change in correlation vector
        """
        occu_i = occu
        delta_corr = np.zeros(self.n_orbit_functions)
        for f in flips:
            occu_f = occu_i.copy()
            occu_f[f[0]] = f[1]
            orbits = self._orbits_by_sites[f[0]]
            delta_corr += self._dcorr_single_flip(occu_f, occu_i,
                                                  self.n_orbit_functions,
                                                  orbits)
            occu_i = occu_f

        return delta_corr

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

    @classmethod
    def from_dict(cls, d):
        """Create a CEProcessor from serialized MSONable dict."""
        return cls(ClusterExpansion.from_dict(d['cluster_expansion']),
                   np.array(d['supercell_matrix']),
                   optimize_indicator=d['indicator'])


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
        self._ewald_inds = inds
        # Lazy set up Ewald Summation since it can be slow
        self.__ewald = ewald_summation

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
        return self._ewald_term.get_ewald_matrix(self.ewald_summation)

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
                                                      self.n_orbit_functions,
                                                      orbits,
                                                      f[0], f[1],
                                                      self.ewald_matrix,
                                                      self._ewald_inds,
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
