"""
This module implements the ClusterExpansion class, which holds the necessary
attributes to fit a set of orbit functions to a dataset of structures and
a corresponding property (most usually energy).
"""

from __future__ import division
import warnings
import numpy as np
from collections.abc import Sequence
from monty.json import MSONable
from pymatgen import Structure
from smol.cofe.configspace.clusterspace import ClusterSubspace
from smol.cofe.wrangler import StructureWrangler
from smol.cofe.regression.estimator import BaseEstimator, CVXEstimator


class ClusterExpansion(MSONable):
    """
    Class for the ClusterExpansion proper. This needs a ClusterSubspace and
    and a corresponding set of ECI

    The main method to use this the predict method to predict the fitted
    property for new structures. This can be used to compare the accuracy
    of the fit with a set of test structures not used in training.

    Although this is purely optional and does not change the class performance,
    it is also recommended you save some information about regression metrics
    such as CV score, test/train rmse, or anything to quantify the "goodness"
    in the metadata dictionary. See for example regression metrics in
    sklearn.metrics for many useful methods to get this quantities.

    This class is also used for Monte Carlo simulations to create a
    CEProcessor that calculates the CE for a fixed supercell size. Before using
    a ClusterExpansion for Monte Carlo you should consider pruning the orbit
    functions with small eci.
    """

    def __init__(self, cluster_subspace, fit_structures, property_vector,
                 feature_matrix=None, supercell_matrices=None, weights=None,
                 ecis=None, estimator=None):
        """
        Args:
            cluster_subspace (ClusterSubspace):
                A StructureWrangler object to provide the fitting data and
                processing
            fit_structures (list):
                list of structures used to fit the cluster expansion
            ecis (array):
                ecis for cluster expansion. Make sure the supplied eci
                correspond to the correlation vector terms (length and order)
        """

        self._subspace = cluster_subspace
        self.ecis = ecis

        self.metadata = {}

    @property
    def prim_structure(self):
        """ Copy of primitive structure which the Expansion is based on """
        return self.subspace.structure.copy()

    @property
    def expansion_structure(self):
        """
        Copy of the expansion structure with only sites included in the
        expansion (i.e. sites with partial occupancies)
        """
        return self.subspace.exp_structure.copy()

    @property
    def subspace(self):
        return self._subspace

    def predict(self, structures, normalized=False):
        """
        Predict the fitted property for a given set of structures.

        Args:
            structures (list or Structure):
                Structures to predict from
            normalized (bool):
                Whether to return the predicted property normalized by
                the prim cell size.
        Returns:
            array
        """
        extensive = not normalized
        if isinstance(structures, Structure):
            corrs = self.subspace.corr_from_structure(structures,
                                                      extensive=extensive)
        else:
            corrs = [self.subspace.corr_from_structure(structure,
                                                       extensive=extensive)
                     for structure in structures]

        return self.estimator.predict(np.array(corrs))

    # This needs further testing. For out-of-training structures
    # the predictions do not always match with those using the original eci
    # with which the fit was done.
    def convert_eci(self, new_basis, orthonormal=False):
        """
        Numerically converts the eci of the cluster expansion to eci in a
        new basis.

        Args:
            new_basis (str):
                name of basis to convert coefficients into.
            orthonormal (bool):
                option to make new basis orthonormal

        Returns: coefficients converted into new_basis
            array
        """
        subspace = self.subspace.copy()
        subspace.change_site_bases(new_basis, orthonormal=orthonormal)
        feature_matrix = np.array([subspace.corr_from_structure(s, m)
                                   for s, m in zip(self._structures,
                                                   self._scmatrices)])
        C = np.matmul(self.feature_matrix.T,
                      np.linalg.pinv(feature_matrix.T))

        return np.matmul(C.T, self.ecis)

    def prune(self, threshold=1E-5):
        """
        Remove ECI's (fitting parameters) and orbits in the ClusterSubspaces
        that have ECI/parameter values smaller than the given threshold.

        This will change the fits error metrics (ie RMSE) a little, but it
        should not be much. If they change a lot then the threshold used is
        probably to high and important ECI are being pruned.

        This will not re-fit the ClusterExpansion. Note that if you re-fit
        after pruning the ECI will probably change and hence also the fit
        performance.
        """

        if self.ecis is None:
            raise RuntimeError('ClusterExpansion has no ECIs. Cannot prune.')

        bit_ids = [i for i, eci in enumerate(self.ecis)
                   if abs(eci) < threshold]
        self.subspace.remove_orbit_bit_combos(bit_ids)

        # Update necessary attributes
        ids_compliment = list(set(range(len(self.ecis))) - set(bit_ids))
        self.estimator.coef_ = self.estimator.coef_[ids_compliment]
        self.ecis = self.estimator.coef_
        self._feature_matrix = self.feature_matrix[:, ids_compliment]
        self._rmse, self._mae, self._maxerr = None, None, None

    def __str__(self):
        corr = np.zeros(self.subspace.n_bit_orderings)
        corr[0] = 1  # zero point cluster
        # This might need to be redefined to take "expectation" using measure
        feature_avg = np.average(self.feature_matrix, axis=0)
        feature_std = np.std(self.feature_matrix, axis=0)
        s = 'ClusterExpansion:\n    Prim Composition: ' \
            f'{self.prim_structure.composition} Num fit structures: ' \
            f'{len(self.property_vector)} ' \
            f'Num orbit functions: {self.subspace.n_bit_orderings}\n'
        ecis = len(corr)*[0.0, ] if self.ecis is None else self.ecis
        s += f'    [Orbit]  id: {str(0):<3}\n'
        s += '        bit       eci\n'
        s += f'        {"[X]":<10}{ecis[0]:<4.3}\n'
        for orbit in self.subspace.iterorbits():
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

    # TODO save the estimator and parameters?
    @classmethod
    def from_dict(cls, d):
        """
        Creates ClusterExpansion from serialized MSONable dict
        """
        structures = [Structure.from_dict(ds) for ds in d['fit_structures']]
        return cls(ClusterSubspace.from_dict(d['cluster_subspace']),
                   fit_structures=structures,
                   feature_matrix=np.array(d['feature_matrix']),
                   property_vector=np.array(d['property_vector']),
                   ecis=d['ecis'])

    def as_dict(self):
        """
        Json-serialization dict representation

        Returns:
            MSONable dict
        """
        structures = [struct.as_dict() for struct in self._structures]
        d = {'@module': self.__class__.__module__,
             '@class': self.__class__.__name__,
             'cluster_subspace': self.subspace.as_dict(),
             'fit_structures': structures,
             'feature_matrix': self.feature_matrix.tolist(),
             'property_vector': self.property_vector.tolist(),
             'estimator': self.estimator.__class__.__name__,
             'ecis': self.ecis.tolist(),
             'metadata' : self.metadata}
        return d
