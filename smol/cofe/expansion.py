from __future__ import division
import warnings
import numpy as np
from .utils import NotFittedError
from .regression.estimator import BaseEstimator


class ClusterExpansion(object):
    """
    Class for the ClusterExpansion proper needs a datawrangler to supply fitting data and an estimator to
    provide the fitting method.
    This is the class that is used to predict as well (i.e. to use in Monte Carlo and beyond)
    """

    def __init__(self, datawrangler, estimator=None, max_dielectric=None, ecis=None):
        """
        Fit ECI's to obtain a cluster expansion. This init function takes in all possible arguments,
        but its much simpler to use one of the factory classmethods below,
        e.g. EciGenerator.unweighted

        Args:
            datawrangler: A StructureWrangler object to provide the fitting data and processing
            max_dielectric: constrain the dielectric constant to be positive and below the
                supplied value (note that this is also affected by whether the primitive
                cell is the correct size)
            estimator: Estimator or sklearn model
        """
        self.wrangler = datawrangler
        self.estimator = estimator
        self.max_dielectric = max_dielectric
        self.ecis = ecis

        if self.estimator is None:
            if self.ecis is None:
                raise AttributeError('No estimator or ecis were given. One of them needs to be provided')
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
            warnings.warn(msg)

        if self.max_dielectric is not None:
            if self.wrangler.cs.use_ewald is False:
                warnings.warn('The StructureWrangler.use_ewald is False can not constrain by max_dieletric'
                                ' This will be ignored', RuntimeWarning)
                return

            if self.wrangler.cs.use_inv_r:
                warnings.warn('Cant use inv_r with max dielectric. This has not been implemented yet. '
                               'inv_r will be ignored.', RuntimeWarning)

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

#TODO make these MSONable?
    @classmethod
    def from_dict(cls, d):
        raise NotImplementedError('needs to be properly implemented')
        #return cls(cluster_expansion=ClusterExpansion.from_dict(d['cluster_expansion']),
         #          structures=[Structure.from_dict(s) for s in d['structures']],
          #         energies=np.array(d['energies']), max_dielectric=d.get('max_dielectric'),
           #        max_ewald=d.get('max_ewald'), supercell_matrices=d['supercell_matrices'],
            #       mu=d.get('mu'), ecis=d.get('ecis'), feature_matrix=d.get('feature_matrix'),
             #      solver=d.get('solver', 'cvxopt_l1'), weights=d['weights'])

    def as_dict(self):
        raise NotImplementedError('needs to be properly implemented')
        #return {'cluster_expansion': self.wrangler.cs.as_dict(),
         #       'structures': [s.as_dict() for s in self.structures],
          #      'energies': self.energies.tolist(),
           #     'supercell_matrices': [cs.supercell_matrix.tolist() for cs in self.supercells],
            #    'max_dielectric': self.max_dielectric,
             #   'max_ewald': self.wrangler.max_ewald,
              #  'mu': self.mu,
               # 'ecis': self.ecis.tolist(),
                #'feature_matrix': self.feature_matrix.tolist(),
                #'weights': self.weights.tolist(),
                #'solver': self.estimator,
                #'@module': self.__class__.__module__,
                #'@class': self.__class__.__name__}
