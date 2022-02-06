"""Implementation of CE processor class for a fixed size super cell.

If you are using a Hamiltonian with an Ewald summation electrostatic term, you
should use the CompositeProcessor with a CEProcessor and an EwaldProcessor
class to handle changes in the electrostatic interaction energy.
"""

__author__ = "Luis Barroso-Luque"

import numpy as np
from collections import defaultdict
from smol.cofe import ClusterSubspace
from smol.moca.processor.base import Processor
from src.cemc_utils import (corr_from_occupancy, factors_from_occupancy,
                            delta_corr_single_flip, delta_factors_single_flip,
                            indicator_delta_corr_single_flip)


class CEProcessor(Processor):
    """CEProcessor class to use a ClusterExpansion in MC simulations.

    A CE processor is optimized to compute correlation vectors and local
    changes in correlation vectors. This class allows the use a cluster
    expansion Hamiltonian to run Monte Carlo based simulations.

    A processor allows an ensemble class to generate a Markov chain
    for sampling thermodynamic properties from a cluster expansion
    Hamiltonian.

    Attributes:
        optimize_indicator (bool):
            If true the local correlation update function specialized for
            indicator bases is used. This should only be used when the
            expansion was fit with an indicator basis and no additional
            normalization.
        coefs (ndarray):
            Fitted coefficients from the cluster expansion.
        n_corr_functions (int):
            Total number of orbit basis functions (correlation functions).
            This includes all possible labellings/orderings for all orbits.
            Same as :code:`ClusterSubspace.n_bit_orderings`.
    """

    def __init__(self, cluster_subspace, supercell_matrix, coefficients,
                 optimize_indicator=False):
        """Initialize a CEProcessor.

        Args:
            cluster_subspace (ClusterSubspace):
                A cluster subspace
            supercell_matrix (ndarray):
                An array representing the supercell matrix with respect to the
                Cluster Expansion prim structure.
            coefficients (ndarray):
                Fit coefficients for the represented cluster expansion.
            optimize_indicator (bool):
                When using an indicator basis, sets the function to compute
                correlation differences to the indicator optimized function.
                This can make MC steps faster.
                Make sure your cluster expansion was indeed fit with an
                indicator basis set, otherwise your MC results are no good.
        """
        super().__init__(cluster_subspace, supercell_matrix, coefficients)

        self.n_corr_functions = self.cluster_subspace.num_corr_functions
        if len(coefficients) != self.n_corr_functions:
            raise ValueError(
                f'The provided coeffiecients are not the right length. '
                f'Got {len(coefficients)} coefficients, the length must be '
                f'{self.n_corr_functions} based on the provided cluster '
                f'subspace.')

        # set the dcorr_single_flip function
        self.optimize_indicator = optimize_indicator
        self._dcorr_single_flip = indicator_delta_corr_single_flip \
            if optimize_indicator \
            else delta_corr_single_flip

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
        # Prepare necssary information for local updates
        mappings = self._subspace.supercell_orbit_mappings(supercell_matrix)
        for cluster_indices, orbit in zip(mappings, self._subspace.orbits):
            self._orbit_list.append(
                (orbit.bit_id, orbit.flat_tensor_indices,
                 orbit.flat_correlation_tensors, cluster_indices)
            )
            for site_ind in np.unique(cluster_indices):
                in_inds = np.any(cluster_indices == site_ind, axis=-1)
                ratio = len(cluster_indices) / np.sum(in_inds)
                # TODO check runtime and scaling vs new implementations
                #  and if not worth it just deprecate optimize_indicator
                if optimize_indicator:
                    self._orbits_by_sites[site_ind].append(
                        (orbit.bit_id, ratio, orbit.bit_combo_array,
                         orbit.bit_combo_inds, orbit.bases_array,
                         cluster_indices[in_inds])
                    )
                else:
                    self._orbits_by_sites[site_ind].append(
                        (orbit.bit_id, ratio, orbit.flat_tensor_indices,
                         orbit.flat_correlation_tensors,
                         cluster_indices[in_inds])
                    )

    def compute_feature_vector(self, occupancy):
        """Compute the correlation vector for a given occupancy string.

        The correlation vector in this case is normalized per supercell. In
        other works it is extensive corresponding to the size of the supercell.

        Args:
            occupancy (ndarray):
                encoded occupation array

        Returns:
            array: correlation vector
        """
        return corr_from_occupancy(occupancy, self.n_corr_functions,
                                   self._orbit_list) * self.size

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
        occu_i = occupancy
        delta_corr = np.zeros(self.n_corr_functions)
        for f in flips:
            occu_f = occu_i.copy()
            occu_f[f[0]] = f[1]
            site_orbit_list = self._orbits_by_sites[f[0]]
            delta_corr += self._dcorr_single_flip(
                occu_f, occu_i, self.n_corr_functions, site_orbit_list)
            occu_i = occu_f

        return delta_corr * self.size

    def as_dict(self) -> dict:
        """
        Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        d = super().as_dict()
        d['optimize_indicator'] = self.optimize_indicator
        return d

    @classmethod
    def from_dict(cls, d):
        """Create a CEProcessor from serialized MSONable dict."""
        return cls(ClusterSubspace.from_dict(d['cluster_subspace']),
                   np.array(d['supercell_matrix']),
                   coefficients=np.array(d['coefficients']),
                   optimize_indicator=d['optimize_indicator'])


class OrbitDecompositionProcessor(Processor):
    """OrbitDecompositionProcessor ClusterExpansion in MC simulations.

    An OrbitDecompositionProcessor is similar to a ClusterExpansionProcessor
    however it computes properties and changes of properties using orbit
    factors rather than correlation functions and ECI.

    For a given ClusterExpansion the results from sampling using this class
    should be the exact same as using a ClusterExpansionProcessor (ie the
    samples are from the same ensemble).

    A few practical and important differences though, the featyre vectors
    that are used and sampled are the orbit factors for each occupancy
    as oposed to the correlation vectors (so if you need correlation vectors
    for further analysis you're probably better off with a standard
    ClusterExpansionProcessor

    However if you don't need correlation vectors, then this is probably
    more efficient and more importantly the scaling complexity with model size
    is in terms of number of orbits (instead of number of corr functions).
    This can give substantial speedups when doing MC calculations of complex
    high component systems.
    """

    def __init__(self, cluster_subspace, supercell_matrix,
                 orbit_factor_tensors, offset):
        """Initialize an OrbitDecompositionProcessor.

        Args:
            cluster_subspace (ClusterSubspace):
                A cluster subspace
            supercell_matrix (ndarray):
                An array representing the supercell matrix with respect to the
                Cluster Expansion prim structure.
            orbit_factor_tensors (sequence of ndarray):
                Sequence of ndarray where each array corresponds to the
                orbit factor tensor. These should be in the same order as their
                corresponding orbits in the given cluster_subspace
            offset (float):
                coefficient/ECI for empty term or equivalently factor for
                empty term
        """
        if len(orbit_factor_tensors) != cluster_subspace.num_orbits - 1:
            raise ValueError(
                f'The number of orbit factor tensors must match the number '
                f' of orbits in the subspace. Got {len(orbit_factor_tensors)} '
                f'orbit factors, but need {cluster_subspace.num_orbits - 1} '
                f' for the given cluster_subspace.')

        #  the orbit multiplicities play the role of coefficients here.
        super().__init__(cluster_subspace, supercell_matrix,
                         coefficients=cluster_subspace.orbit_multiplicities)

        self.n_orbits = self.cluster_subspace.num_orbits
        self.offset = offset
        self._fac_tensors = orbit_factor_tensors

        # List of orbit information and supercell site indices to compute corr
        self._orbit_list = []
        # Dictionary of orbits by site index and information
        # necessary to compute local changes in correlation vectors from flips
        self._orbits_by_sites = defaultdict(list)
        # Prepare necssary information for local updates
        mappings = self._subspace.supercell_orbit_mappings(supercell_matrix)
        for cluster_indices, orbit_factor, orbit in zip(
                mappings, orbit_factor_tensors, self._subspace.orbits):
            flat_factor_tensor = np.ravel(orbit_factor, order='C')
            self._orbit_list.append(
                (orbit.flat_tensor_indices, flat_factor_tensor,
                 cluster_indices)
            )
            for site_ind in np.unique(cluster_indices):
                in_inds = np.any(cluster_indices == site_ind, axis=-1)
                ratio = len(cluster_indices) / np.sum(in_inds)
                self._orbits_by_sites[site_ind].append(
                    (orbit.id, ratio, orbit.flat_tensor_indices,
                     flat_factor_tensor, cluster_indices[in_inds])
                )

    def compute_feature_vector(self, occupancy):
        """Compute the orbit factor vector for a given occupancy string.

        The orbit factor vector in this case is normalized per supercell.

        Args:
            occupancy (ndarray):
                encoded occupation array

        Returns:
            array: correlation vector
        """
        return factors_from_occupancy(occupancy, self.n_orbits, self.offset,
                                      self._orbit_list) * self.size

    def compute_feature_vector_change(self, occupancy, flips):
        """
        Compute the change in the orbit factor vector from a list of flips.

        The orbit factor vector in this case is normalized per supercell.

        Args:
            occupancy (ndarray):
                encoded occupancy string
            flips (list of tuple):
                list of tuples with two elements. Each tuple represents a
                single flip where the first element is the index of the site
                in the occupancy string and the second element is the index
                for the new species to place at that site.

        Returns:
            array: change in orbit factor vector
        """
        occu_i = occupancy
        delta_factors = np.zeros(self.n_orbits)
        for f in flips:
            occu_f = occu_i.copy()
            occu_f[f[0]] = f[1]
            site_orbit_list = self._orbits_by_sites[f[0]]
            delta_factors += delta_factors_single_flip(
                occu_f, occu_i, self.n_orbits, site_orbit_list)
            occu_i = occu_f

        return delta_factors * self.size

    def as_dict(self) -> dict:
        """
        Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        d = super().as_dict()
        d['offset'] = self.offset
        d['orbit_factor_tensors'] = [factor.tolist() for factor
                                     in self._fac_tensors]
        return d

    @classmethod
    def from_dict(cls, d):
        """Create a CEProcessor from serialized MSONable dict."""
        factor_tensors = tuple(
            np.array(factor) for factor in d['orbit_factor_tensors']
        )
        return cls(ClusterSubspace.from_dict(d['cluster_subspace']),
                   np.array(d['supercell_matrix']), factor_tensors,
                   d['offset'])
