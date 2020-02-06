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
from smol.exceptions import NotFittedError


class ClusterExpansion(MSONable):
    """
    Class for the ClusterExpansion proper needs a structure_wrangler to supply
    fitting data and an estimator to provide the fitting method.
    This is the class that is used to predict as well.
    (i.e. to use in Monte Carlo and beyond)
    """

    def __init__(self, cluster_subspace, fit_structures, property_vector,
                 feature_matrix=None, supercell_matrices=None, weights=None,
                 ecis=None, estimator=None):
        """
        Represents a cluster expansion. The main methods to use this class are
        the fit method to fit the cluster expansion using the provided
        structures (and or feature matrix) and the predict method to predict
        the fitted property to new structures. The ClusterExpansion also
        contains a few regression metrics methods to check the quality of the
        fit.

        Args:
            cluster_subspace (ClusterSubspace):
                A StructureWrangler object to provide the fitting data and
                processing
            fit_structures (list):
                list of structures used to fit the cluster expansion
            property_vector (np.array):
                1D array with the value of the property to fit to corresponding
                to the structures in the feature matrix.
            feature_matrix (np.array): optional
                2D array with features, the correlation vectors for each
                structure in the training data. If this has already been
                computed, fitting the Expansion will be A LOT faster.
                If not provided then it will be computed using the given
                structures. Make sure this was computed with an the same
                cluster_subspace provided.
            supercell_matrices (list):
                list of supercell matrices relating subspaces prim structure to
                the corresponding fit structures. Providing these will make
                computing things much faster since the structures no longer
                need to be matched to the prim structure
            weights (np.array): optional
                1D array of weights for each data point (structure) in feature
                matrix
            ecis (array): optional
                ecis for cluster expansion. This should only be used if the
                expansion was already fitted or the eci where obtained
                externally by some other means. Make sure the supplied eci
                correspond to the correlation vector terms (length and order)
            estimator: optional
                Estimator class with fit and predict functionality. See
                smol.cofe.regression.estimator for details. Either ECIs or an
                estimator must be provided.
        """

        # Check the shape of all input
        if len(fit_structures) != len(property_vector):
            raise ValueError(f'Number of provided fit structures '
                             f'{len(fit_structures)} does not '
                             f'correspond to property vector of shape '
                             f'{property_vector.shape}')
        if weights is not None and len(weights) != len(property_vector):
            raise ValueError(f'Provided weights of length '
                             f'{len(weights)} do not match the '
                             f'{len(fit_structures)} fit structures provided.')
        if supercell_matrices is not None:
            if len(supercell_matrices) != len(property_vector):
                raise ValueError(f'Provided list of supercell matrices of '
                                 f'length {len(supercell_matrices)} does not '
                                 f'correspond to the {len(fit_structures)} fit'
                                 f' structures provided.')
        else:
            supercell_matrices = len(fit_structures)*[None, ]

        self._subspace = cluster_subspace
        self._structures = fit_structures
        self._feature_matrix = feature_matrix
        self._scmatrices = supercell_matrices
        self._property_vector = property_vector
        self._weights = weights
        self.estimator = estimator
        self.ecis = ecis

        # Fit metrics
        self._maxerr = None
        self._mae = None
        self._rmse = None

        if self.estimator is None:
            if self.ecis is None:
                raise AttributeError('No estimator or ECIs were given. '
                                     'One of them needs to be provided.')
            self.estimator = BaseEstimator()
            self.estimator.coef_ = self.ecis

    @classmethod
    def from_radii(cls, structure, radii, ltol=0.2, stol=0.1, angle_tol=5,
                   supercell_size='volume', basis='indicator',
                   orthonormal=False, external_terms=None, estimator=None,
                   ecis=None, data=None, verbose=False, weights=None):
        """
        This convenience method creates a ClusterExpansion in one go (with no
        need to create the underlying objects necessary) This is the quickest
        and easiest way to get a ClusterExpansion up and running.

        Args:
            structure:
                disordered structure to build a cluster expansion for.
                Typically the primitive cell
            radii:
                dict of {cluster_size: max_radius}. Radii should be strictly
                decreasing. Typically something like {2:5, 3:4}
            ltol, stol, angle_tol, supercell_size: parameters to pass through
                to the StructureMatcher. Structures that don't match to the
                primitive cell under these tolerances won't be included in the
                expansion. Easiest option for supercell_size is usually to use
                a species that has a constant amount per formula unit.
            basis (str):
                a string specifying the site basis functions
            orthonormal (bool):
                whether to enforce an orthonormal basis. From the current
                available bases only the sinusoid basis is orthogonal out
                of the box for any number of species. Legendre and Chebyshev
                are orthogonal for only 2 species out of the box.
            external_terms (object):
                any external terms to add to the cluster subspace
                Currently only an EwaldTerm.
            estimator: optional
                Estimator or sklearn model. Needs to have a fit and predict
                method, fitted coefficients must be stored in _coeffs
                attribute (usually these are the ECI).
            ecis (array):
                ecis for cluster expansion. This should only be used if the
                expansion was already fitted. Make sure the supplied eci
                correspond to the correlation vector terms (length and order)
            data (list):
                list of (structure, property) data
            verbose (bool):
                if True then print structures that fail in StructureMatcher
            weights (str, list/tuple or array):
                str specifying type of weights (i.e. 'hull') OR
                list/tuple with two elements (name, kwargs) were name specifies
                the type of weights as above, and kwargs are a dict of
                keyword arguments to obtain the weights OR
                array directly specifying the weights
        Returns:
            ClusterExpansion (not automatically fitted)
        """

        subspace = ClusterSubspace.from_radii(structure, radii, ltol, stol,
                                              angle_tol, supercell_size, basis,
                                              orthonormal)
        if external_terms is not None:
            # at some point we should loop through this if more than 1 term
            kwargs = {}
            if isinstance(external_terms, Sequence):
                external_terms, kwargs = external_terms
            subspace.add_external_term(external_terms, **kwargs)

        # Create the wrangler to obtain training data
        wrangler = StructureWrangler(subspace)
        if data is not None:
            wrangler.add_data(data, verbose=verbose, weights=weights)
        elif isinstance(weights, str):
            if weights not in wrangler.get_weights.keys():
                raise AttributeError(f'Weight str provided {weights} is not'
                                     f'valid. Choose one of '
                                     f'{wrangler.weights.keys()}')
            wrangler.weight_type = weights
        if estimator is None and ecis is None:
            estimator = CVXEstimator()

        return cls(subspace, fit_structures=wrangler.refined_structures,
                   feature_matrix=wrangler.feature_matrix,
                   property_vector=wrangler.normalized_properties,
                   weights=wrangler.weights, estimator=estimator, ecis=ecis)

    @classmethod
    def from_structure_wrangler(cls, structure_wrangler, estimator=None,
                                ecis=None):
        """
        Convenience method to construct a cluster expansion from a structure
        wrangler. This is the recommended method to create a cluster expansion
        if more control over the cluster subspace and further data
        pre-processing needs to be done to the fit structures.

        Args:
            structure_wrangler (StructureWrangler):
            estimator (object):
                An estimator instance to fit the expansion
            ecis (array):
                ecis if already obtained. Either an estimator or ecis need to
                be provided.

        Returns:
            ClusterExpansion
        """
        if estimator is None and ecis is None:
            estimator = CVXEstimator()

        return cls(structure_wrangler.cluster_subspace,
                   fit_structures=structure_wrangler.refined_structures,
                   property_vector=structure_wrangler.normalized_properties,
                   feature_matrix=structure_wrangler.feature_matrix,
                   supercell_matrices=structure_wrangler.supercell_matrices,
                   weights=structure_wrangler.weights, estimator=estimator,
                   ecis=ecis)

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
    def feature_matrix(self):
        if self._feature_matrix is None:
            matrix = [self.subspace.corr_from_structure(s, m)
                      for s, m in zip(self._structures, self._scmatrices)]
            self._feature_matrix = np.array(matrix)
        return self._feature_matrix.copy()

    @property
    def property_vector(self):
        if self._property_vector is not None:
            return self._property_vector.copy()

    @property
    def weights(self):
        if self._weights is not None:
            return self._weights.copy()

    @property
    def subspace(self):
        return self._subspace

    @property
    def max_error(self):
        """Maximum residual error of fit"""
        if self._maxerr is None:
            pred = self.estimator.predict(self.feature_matrix)
            self._maxerr = np.max(np.abs(self.property_vector - pred))
        return self._maxerr

    @property
    def mean_absolute_error(self):
        """(Weighted) mean absolute error of fit"""
        if self._mae is None:
            pred = self.estimator.predict(self.feature_matrix)
            self._mae = np.average(np.abs(self.property_vector - pred),
                                   weights=self.weights)
        return self._mae

    @property
    def root_mean_squared_error(self):
        """(Weighted) root mean squared error of fit"""
        if self._rmse is None:
            pred = self.estimator.predict(self.feature_matrix)
            mse = np.average((self.property_vector - pred)**2,
                             weights=self.weights)
            self._rmse = np.sqrt(mse)
        return self._rmse

    def fit(self, *args, **kwargs):
        """
        Fits the cluster expansion using the given estimator's fit function
        args, kwargs are the arguments and keyword arguments taken by the
        Estimator.fit function
        """
        A_in = self.feature_matrix
        y_in = self.property_vector

        if self.weights is not None:
            self.estimator.fit(A_in, y_in, self.weights,
                               *args, **kwargs)
        else:
            self.estimator.fit(A_in, y_in, *args, **kwargs)

        try:
            self.ecis = self.estimator.coef_
        except AttributeError:
            warnings.warn(f'The provided estimator does not provide fit '
                          f'coefficients for ECIS: {self.estimator}')

        # reset fit metrics
        self._maxerr, self._mae, self._rmse = None, None, None

    def predict(self, structures, normalized=False):
        """
        Predict the fitted property for a given set of structures.

        Args:
            structures (list or Structure):
                Structures to predict from
            normalized (bool):
                Whether to return the predicted property normalized by
                supercell size.
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

    # TODO implement this and add test.
    def prune(self, threshold=1E-5):
        """
        Remove ECI's and orbits in the ClusterSubspaces that have ECI values
        smaller than the given threshold
        """
        pass

    # TODO change this to  __str__
    def print_ecis(self):
        if self.ecis is None:
            raise NotFittedError('This ClusterExpansion has no ECIs available.'
                                 ' If it has not been fitted yet, run '
                                 'ClusterExpansion.fit to do so.'
                                 'Otherwise you may have chosen an estimator '
                                 'that does not provide them:'
                                 f'{self.estimator}.')

        corr = np.zeros(self.subspace.n_bit_orderings)
        corr[0] = 1  # zero point cluster
        cluster_std = np.std(self.feature_matrix, axis=0)
        for orbit in self.subspace.iterorbits():
            print(orbit, len(orbit.bits) - 1, orbit.orb_b_id)
            print('bit    eci    cluster_std    eci*cluster_std')
            for i, bits in enumerate(orbit.bit_combos):
                eci = self.ecis[orbit.orb_b_id + i]
                c_std = cluster_std[orbit.orb_b_id + i]
                print(bits, eci, c_std, eci * c_std)
        print(self.ecis)

    def __str__(self):
        corr = np.zeros(self.subspace.n_bit_orderings)
        corr[0] = 1  # zero point cluster
        # This might need to be redefined to take "expectation" using measure
        feature_avg = np.average(self.feature_matrix, axis=0)
        feature_std = np.std(self.feature_matrix, axis=0)
        s = f'ClusterExpansion:\n    Prim Composition: ' \
            f'{self.prim_structure.composition} Num fit structures: ' \
            f'{len(self.property_vector)} ' \
            f'Num orbit functions: {self.subspace.n_bit_orderings}\n'
        ecis = len(corr)*[None, ] if self.ecis is None else self.ecis
        s += f'    [Orbit]  id: {str(0):<3}\n'
        s += f'        bit       eci\n'
        s += f'        {"[X]":<10}{ecis[0]:<4.3}\n'
        for orbit in self.subspace.iterorbits():
            s += f'    [Orbit]  id: {orbit.orb_b_id:<3} size: ' \
                 f'{len(orbit.bits):<3} radius: {orbit.radius:<4.3}\n'
            s += f'        bit       eci     feature avg  feature std  ' \
                 f'eci*std\n'
            for i, bits in enumerate(orbit.bit_combos):
                eci = ecis[orbit.orb_b_id + i]
                f_avg = feature_avg[orbit.orb_b_id + i]
                f_std = feature_std[orbit.orb_b_id + i]
                s += f'        {str(bits[0]):<10}{eci:<8.3f}{f_avg:<13.3f}' \
                     f'{f_std:<13.3f}{eci*f_std:<.3f}\n'
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
             'ecis': self.ecis.tolist()}
        return d
