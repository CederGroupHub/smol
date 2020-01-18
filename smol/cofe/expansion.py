"""
This module implements the ClusterExpansion class, which holds the necessary
attributes to fit a set of orbit functions to a dataset of structures and
a corresponding property (most usually energy).
"""

from __future__ import division
import warnings
import numpy as np
from monty.json import MSONable
from smol.cofe.utils import NotFittedError
from smol.cofe.wrangler import StructureWrangler
from .regression.estimator import BaseEstimator


class ClusterExpansion(MSONable):
    """
    Class for the ClusterExpansion proper needs a structurewrangler to supply
    fitting data and an estimator to provide the fitting method.
    This is the class that is used to predict as well.
    (i.e. to use in Monte Carlo and beyond)
    """

    def __init__(self, structwrangler, estimator=None, ecis=None):
        """
        Represents a cluster expansion. The main methods to use this class are
        the fit and predict

        Args:
            structwrangler (StructureWrangler):
                A StructureWrangler object to provide the fitting data and
                processing
            max_dielectric (float):
                Constrain the dielectric constant to be positive and below the
                supplied value (note that this is also affected by whether the
                primitive cell is the correct size)
            estimator:
                Estimator or sklearn model. Needs to have a fit and predict
                method, fitted coefficients must be stored in _coeffs
                attribute (usually these are the ECI).
        """

        self.wrangler = structwrangler
        self.estimator = estimator
        self.ecis = ecis

        if self.estimator is None:
            if self.ecis is None:
                raise AttributeError('No estimator or ECIs were given. '
                                     'One of them needs to be provided')
            self.estimator = BaseEstimator()
            self.estimator.coef_ = self.ecis

    def fit(self, *args, **kwargs):
        """
        Fits the cluster expansion using the given estimator's fit function
        args, kwargs are the arguments and keyword arguments taken by the
        Estimator.fit function
        """
        A_in = self.wrangler.feature_matrix.copy()
        y_in = self.wrangler.normalized_properties.copy()

        if self.wrangler.weight_type is not None:
            self.estimator.fit(A_in, y_in,
                               self.wrangler.weights,
                               *args, **kwargs)
        else:
            self.estimator.fit(A_in, y_in, *args, **kwargs)

        try:
            self.ecis = self.estimator.coef_
        except AttributeError:
            warnings.warn(f'The provided estimator does not provide fit '
                          f'coefficients for ECIS: {self.estimator}')
            return

    def predict(self, structures, normalized=False):
        structures = structures if type(structures) == list else [structures]
        corrs = []
        for structure in structures:
            corr, size = self.wrangler.cs.corr_from_structure(structure,
                                                              return_size=True)
            if not normalized:
                corr *= size
            corrs.append(corr)

        return self.estimator.predict(np.array(corrs))

    # TODO change this to  __str__
    def print_ecis(self):
        if self.ecis is None:
            raise NotFittedError('This ClusterExpansion has no ECIs available.'
                                 'If it has not been fitted yet, run'
                                 'ClusterExpansion.fit to do so.'
                                 'Otherwise you may have chosen an estimator'
                                 'that does not provide them:'
                                 f'{self.estimator}.')

        corr = np.zeros(self.wrangler.cs.n_bit_orderings)
        corr[0] = 1  # zero point cluster
        cluster_std = np.std(self.wrangler.feature_matrix, axis=0)
        for orbit in self.wrangler.cs.iterorbits():
            print(orbit, len(orbit.bits) - 1, orbit.orb_b_id)
            print('bit    eci    cluster_std    eci*cluster_std')
            for i, bits in enumerate(orbit.bit_combos):
                eci = self.ecis[orbit.orb_b_id + i]
                c_std = cluster_std[orbit.orb_b_id + i]
                print(bits, eci, c_std, eci * c_std)
        print(self.ecis)

    # TODO save the estimator and parameters?
    @classmethod
    def from_dict(cls, d):
        """
        Creates ClusterExpansion from serialized MSONable dict
        """
        return cls(StructureWrangler.from_dict(d['wrangler']), ecis=d['ecis'])

    def as_dict(self):
        """
        Json-serialization dict representation

        Returns:
            MSONable dict
        """

        d = {'@module': self.__class__.__module__,
             '@class': self.__class__.__name__,
             'wrangler': self.wrangler.as_dict(),
             'estimator': self.estimator.__class__.__name__,
             'ecis': self.ecis.tolist()}
        return d
