from __future__ import division
import warnings
import numpy as np
from monty.json import MSONable
from .utils import NotFittedError
from . import StructureWrangler
from .regression.estimator import BaseEstimator

# TODO think how to do this max_dielectric thing...


class ClusterExpansion(MSONable):
    """
    Class for the ClusterExpansion proper needs a structurewrangler to supply fitting data and
    an estimator to provide the fitting method.
    This is the class that is used to predict as well (i.e. to use in Monte Carlo and beyond)
    """

    def __init__(self, structwrangler, estimator=None, max_dielectric=None, ecis=None):
        """
        Fit ECI's to obtain a cluster expansion. This init function takes in all possible arguments,
        but its much simpler to use one of the factory classmethods below,
        e.g. EciGenerator.unweighted

        Args:
            structwrangler (StructureWrangler):
                A StructureWrangler object to provide the fitting data and processing
            max_dielectric (float):
                Constrain the dielectric constant to be positive and below the
                supplied value (note that this is also affected by whether the primitive
                cell is the correct size)
            estimator:
                Estimator or sklearn model. Needs to have a fit and predict method, fitted coefficients
                must be stored in _coeffs attribute (usually these are the ECI).
        """

        self.wrangler = structwrangler
        self.estimator = estimator
        self.max_dielectric = max_dielectric
        self.ecis = ecis

        if self.estimator is None:
            if self.ecis is None:
                raise AttributeError('No estimator or ECIs were given. One of them needs to be provided')
            self.estimator = BaseEstimator()
            self.estimator.coef_ = self.ecis

    def fit(self, *args, **kwargs):
        """
        Fits the cluster expansion using the given estimator's fit function
        args, kwargs are the arguments and keyword arguments taken by the Estimator.fit function
        """
        A_in = self.wrangler.feature_matrix.copy()
        y_in = self.wrangler.normalized_properties.copy()

        self.estimator.fit(A_in, y_in, *args, **kwargs)
        try:
            self.ecis = self.estimator.coef_
        except AttributeError:
            msg = f'The provided estimator does not provide fit coefficients for ECIS: {self.estimator}'
            if self.max_dielectric is not None:
                msg += ' constrain by max dielectric does not work without ECIS. Will Ignore.'
            warnings.warn(msg)
            return

        #TODO make this more modular. its really ugly
        for term, args, kwargs in self.wrangler.cs.external_terms:
            if term.__name__ == 'EwaldTerm':
                if kwargs['use_inv_r']:
                    warnings.warn('The StructureWrangler.use_ewald is False can not constrain by max_dieletric'
                                  ' This will be ignored', RuntimeWarning)
                    return

        if self.ecis[-1] < 1 / self.max_dielectric:
            y_in -= A_in[:, -1] / self.max_dielectric
            A_in[:, -1] = 0
            self.estimator.fit(A_in, y_in, *args, **kwargs)
            self.ecis[-1] = 1 / self.max_dielectric

    def predict(self, structures, normalized=False):
        structures = structures if type(structures) == list else [structures]
        corrs = []
        for structure in structures:
            corr, size = self.wrangler.cs.corr_from_structure(structure, return_size=True)
            if not normalized:
                corr *= size
            corrs.append(corr)

        return self.estimator.predict(np.array(corrs))

    def print_ecis(self):
        if self.ecis is None:
            raise NotFittedError('This ClusterExpansion has no ECIs available.'
                                 'If it has not been fitted yet, run ClusterExpansion.fit to do so.'
                                 'Otherwise you may have chosen an estimator that does not provide them:'
                                 f'{self.estimator}.')

        corr = np.zeros(self.wrangler.cs.n_bit_orderings)
        corr[0] = 1  # zero point cluster
        cluster_std = np.std(self.feature_matrix, axis=0)
        for orbit in self.wrangler.cs.iterorbits():
            print(orbit, len(orbit.bits) - 1, orbit.sc_b_id)
            print('bit    eci    cluster_std    eci*cluster_std')
            for i, bits in enumerate(orbit.bit_combos):
                eci = self.ecis[orbit.sc_b_id + i]
                c_std = cluster_std[orbit.sc_b_id + i]
                print(bits, eci, c_std, eci * c_std)
        print(self.ecis)

    #TODO save the estimator and parameters?
    @classmethod
    def from_dict(cls, d):
        """
        Creates ClusterExpansion from serialized MSONable dict
        """

        return cls(StructureWrangler.from_dict(d['wrangler']), max_dielectric=d['max_dielectric'],
                   ecis=['ecis'])

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
             'max_dielectric': self.max_dielectric,
             'ecis': self.ecis.tolist()}
        return d
