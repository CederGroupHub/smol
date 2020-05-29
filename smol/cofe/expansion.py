"""This module implements the ClusterExpansion class.

A ClusterExpansion holds the necessary attributes to represent a CE and predict
the property for new structures.

The class also allows to prune a CE to remove low importance orbits function
terms and speed up Monte Carlo runs.

Also has numerical ECI conversion to other basis sets, but has not been
strongly tested.
"""

__author__ = "Luis Barroso-Luque"

import numpy as np
from monty.json import MSONable
from smol.cofe.configspace.clusterspace import ClusterSubspace


class ClusterExpansion(MSONable):
    """Class for the ClusterExpansion proper.

    This needs a ClusterSubspace and a corresponding set of ECI from a fit.

    The main method to use this the predict method to predict the fitted
    property for new structures. This can be used to compare the accuracy
    of the fit with a set of test structures not used in training.

    Although this is purely optional and does not change the class performance,
    it is also recommended you save some information about learn metrics
    such as CV score, test/train rmse, or anything to quantify the "goodness"
    in the metadata dictionary. See for example learn metrics in
    sklearn.metrics for many useful methods to get this quantities.

    This class is also used for Monte Carlo simulations to create a
    CEProcessor that calculates the CE for a fixed supercell size. Before using
    a ClusterExpansion for Monte Carlo you should consider pruning the orbit
    functions with small eci.
    """

    def __init__(self, cluster_subspace, ecis, feature_matrix):
        """
        Args:
            cluster_subspace (ClusterSubspace):
                A StructureWrangler object to provide the fitting data and
                processing
            ecis (ndarray):
                ecis for cluster expansion. Make sure the supplied eci
                correspond to the correlation vector terms (length and order)
            feature_matrix (ndarray)
                the feature matrix used in fitting the given eci
        """

        self.ecis = ecis
        self.metadata = {}
        self._subspace = cluster_subspace
        self._feat_matrix = feature_matrix
        self._eci_orbit_ids = None

    @property
    def prim_structure(self):
        """Primitive structure which the Expansion is based on."""
        return self.cluster_subspace.structure

    @property
    def expansion_structure(self):
        """Expansion structure with only sites included in the expansion.
        (i.e. sites with partial occupancies)
        """
        return self.cluster_subspace.exp_structure

    @property
    def cluster_subspace(self):
        return self._subspace

    @property
    def eci_orbit_ids(self):
        """Orbit ids corresponding to each ECI in the Cluster Expansion.

        If the Cluster Expansion includes external terms these are not included
        in the list since they are not associated with any orbit.
        """
        if self._eci_orbit_ids is None:
            self._eci_orbit_ids = [0]
            for orbit in self._subspace.iterorbits():
                self._eci_orbit_ids += orbit.n_bit_orderings*[orbit.id, ]
        return self._eci_orbit_ids

    def predict(self, structure, normalize=False):
        """Predict the fitted property for a given set of structures.

        Args:
            structure (Structure):
                Structures to predict from
            normalize (bool):
                Whether to return the predicted property normalized by
                the prim cell size.
        Returns:
            array
        """

        corrs = self.cluster_subspace.corr_from_structure(structure,
                                                          normalized=normalize)
        return np.dot(np.array(corrs), self.ecis)

    def prune(self, threshold=0):
        """Remove ECI's (fitting parameters) and orbits in the ClusterSubspaces
        that have ECI/parameter values smaller than the given threshold.

        This will change the fits error metrics (ie RMSE) a little, but it
        should not be much. If they change a lot then the threshold used is
        probably to high and important ECI are being pruned.

        This will not re-fit the ClusterExpansion. Note that if you re-fit
        after pruning the ECI will probably change and hence also the fit
        performance.
        """

        bit_ids = [i for i, eci in enumerate(self.ecis)
                   if abs(eci) < threshold]
        self.cluster_subspace.remove_orbit_bit_combos(bit_ids)
        # Update necessary attributes
        ids_compliment = list(set(range(len(self.ecis))) - set(bit_ids))
        self.ecis = self.ecis[ids_compliment]
        self._eci_orbit_ids = None  # reset this
        self._feat_matrix = self._feat_matrix[:, ids_compliment]

    # This needs further testing. For out-of-training structures
    # the predictions do not always match with those using the original eci
    # with which the fit was done.
    def convert_eci(self, new_basis, fit_structures, supercell_matrices,
                    orthonormal=False):
        """Numerically converts the eci of the cluster expansion to eci in a
        new basis.

        Args:
            new_basis (str):
                name of basis to convert coefficients into.
            fit_structures (list):
                list of pymatgen.Structure used to fit the eci
            supercell_matrices (list):
                list of supercell matrices for the corresponding fit structures
            orthonormal (bool):
                option to make new basis orthonormal

        Returns:
            array: coefficients converted into new_basis
        """
        subspace = self.cluster_subspace.copy()
        subspace.change_site_bases(new_basis, orthonormal=orthonormal)
        new_feature_matrix = np.array([subspace.corr_from_structure(s,
                                                                    scmatrix=m)
                                       for s, m in zip(fit_structures,
                                                       supercell_matrices)])
        C = np.matmul(self._feat_matrix.T,
                      np.linalg.pinv(new_feature_matrix.T))
        return np.matmul(C.T, self.ecis)

    def __str__(self):
        """Pretty string for printing."""
        corr = np.zeros(self.cluster_subspace.n_bit_orderings)
        corr[0] = 1  # zero point cluster
        # This might need to be redefined to take "expectation" using measure
        feature_avg = np.average(self._feat_matrix, axis=0)
        feature_std = np.std(self._feat_matrix, axis=0)
        s = 'ClusterExpansion:\n    Prim Composition: ' \
            f'{self.prim_structure.composition}\n Num fit structures: ' \
            f'{self._feat_matrix.shape[0]}\n' \
            f'Num orbit functions: {self.cluster_subspace.n_bit_orderings}\n'
        ecis = len(corr)*[0.0, ] if self.ecis is None else self.ecis
        s += f'    [Orbit]  id: {str(0):<3}\n'
        s += '        bit       eci\n'
        s += f'        {"[X]":<10}{ecis[0]:<4.3}\n'
        for orbit in self.cluster_subspace.iterorbits():
            s += f'    [Orbit]  id: {orbit.bit_id:<3} size: ' \
                 f'{len(orbit.bits):<3} radius: {orbit.radius:<4.3}\n'
            s += '        id    bit       eci     feature avg  feature std  '\
                 'eci*std\n'
            for i, bits in enumerate(orbit.bit_combos):
                eci = ecis[orbit.bit_id + i]
                f_avg = feature_avg[orbit.bit_id + i]
                f_std = feature_std[orbit.bit_id + i]
                s += f'        {orbit.bit_id + i:<6}{str(bits[0]):<10}' \
                     f'{eci:<8.3f}{f_avg:<13.3f}{f_std:<13.3f}' \
                     f'{eci*f_std:<.3f}\n'
        return s

    @classmethod
    def from_dict(cls, d):
        """
        Creates ClusterExpansion from serialized MSONable dict.
        """
        ce = cls(ClusterSubspace.from_dict(d['cluster_subspace']),
                 ecis=np.array(d['ecis']),
                 feature_matrix=np.array(d['feature_matrix']))
        ce.metadata = d['metadata']
        return ce

    def as_dict(self):
        """
        Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        d = {'@module': self.__class__.__module__,
             '@class': self.__class__.__name__,
             'cluster_subspace': self.cluster_subspace.as_dict(),
             'ecis': self.ecis.tolist(),
             'feature_matrix': self._feat_matrix.tolist(),
             'metadata': self.metadata}
        return d
