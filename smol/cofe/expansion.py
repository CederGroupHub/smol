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

    Attributes:
        coefficients (ndarrya): ECIS of the cluster expansion
        metadata (dict): dict to save optional values describing cluster
            expansion. i.e. if it was pruned, any error metrics etc.
    """

    def __init__(self, cluster_subspace, coefficients, feature_matrix):
        """Initialize a ClusterExpansion.

        Args:
            cluster_subspace (ClusterSubspace):
                A clustersubspace representing the subspace over which the
                Cluster Expansion was fit. Must be the same used to create
                the feature matrix.
            coefficients (ndarray):
                coefficients for cluster expansion. Make sure the supplied
                coefficients to the correlation vector terms (length and order)
                These correspond to the ECI x the multiplicity of their orbit.
            feature_matrix (ndarray)
                the feature matrix used in fitting the given coefficients.
        """
        if len(coefficients) != feature_matrix.shape[1]:
            raise AttributeError(f'Feature matrix shape {feature_matrix.shape}'
                                 'does not match length of coefficients '
                                 f'{len(coefficients)}.')
        self.coefs = coefficients
        self.metadata = {}
        self._subspace = cluster_subspace
        self._feat_matrix = feature_matrix
        self._eci = None

    @property
    def eci(self):
        """Get the eci for the cluster expansion.

        This just divides by the corresponding multiplicities. External terms
        will be dropped.
        """
        if self._eci is None:
            mults = [1]  # empty orbit
            for mult, ords in zip(self._subspace.orbit_multiplicities,
                                  self._subspace.orbit_nbit_orderings):
                mults += ords*[mult, ]
            n = len(self._subspace.external_terms)  # check for extra terms
            coefs = self.coefs[:-n] if n else self.coefs
            self._eci = coefs/np.array(mults)
        return self._eci

    @property
    def prim_structure(self):
        """Get primitive structure which the expansion is based on."""
        return self.cluster_subspace.structure

    @property
    def expansion_structure(self):
        """Get expansion structure.

        Prim structure with only sites included in the expansion.
        (i.e. sites with partial occupancies)
        """
        return self.cluster_subspace.exp_structure

    @property
    def cluster_subspace(self):
        """Get cluster subspace."""
        return self._subspace

    @property
    def eci_orbit_ids(self):
        """Get Orbit ids corresponding to each ECI in the Cluster Expansion.

        If the Cluster Expansion includes external terms these are not included
        in the list since they are not associated with any orbit.
        """
        return self._subspace.function_orbit_ids

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
        return np.dot(np.array(corrs), self.coefs)

    def prune(self, threshold=0, with_multiplicity=False):
        """Remove fit coefficients or ECI's with small values.

        Removes ECI's and and orbits in the ClusterSubspaces that have
        ECI/parameter values smaller than the given threshold.

        This will change the fits error metrics (ie RMSE) a little, but it
        should not be much. If they change a lot then the threshold used is
        probably to high and important ECI are being pruned.

        This will not re-fit the ClusterExpansion. Note that if you re-fit
        after pruning the ECI will probably change and hence also the fit
        performance.

        Args:
            threshold (float):
                threshold below which to remove.
            with_multiplicity (bool):
                if true threshold is applied to the ECI proper, otherwise to
                the fit coefficients
        """
        coefs = self.eci if with_multiplicity else self.coefs
        bit_ids = [i for i, coef in enumerate(coefs)
                   if abs(coef) < threshold]
        self.cluster_subspace.remove_orbit_bit_combos(bit_ids)
        # Update necessary attributes
        ids_complement = list(set(range(len(self.coefs))) - set(bit_ids))
        ids_complement.sort()
        self.coefs = self.coefs[ids_complement]
        self._feat_matrix = self._feat_matrix[:, ids_complement]
        self._eci = None  # Reset

    # This needs further testing. For out-of-training structures
    # the predictions do not always match with those using the original eci
    # with which the fit was done.
    def convert_eci(self, new_basis, fit_structures, supercell_matrices,
                    orthonormal=False):
        """Numerically convert given eci to eci in a new basis.

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
        return np.matmul(C.T, self.coefs)

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
        ecis = len(corr)*[0.0, ] if self.coefs is None else self.coefs
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
        """Create ClusterExpansion from serialized MSONable dict."""
        ce = cls(ClusterSubspace.from_dict(d['cluster_subspace']),
                 coefficients=np.array(d['coefs']),
                 feature_matrix=np.array(d['feature_matrix']))
        ce.metadata = d['metadata']
        return ce

    def as_dict(self):
        """
        Get Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        d = {'@module': self.__class__.__module__,
             '@class': self.__class__.__name__,
             'cluster_subspace': self.cluster_subspace.as_dict(),
             'coefs': self.coefs.tolist(),
             'feature_matrix': self._feat_matrix.tolist(),
             'metadata': self.metadata}
        return d
