"""This module implements the ClusterExpansion class.

A ClusterExpansion holds the necessary attributes to represent a CE and predict
the property for new structures.

The class also allows to prune a CE to remove low importance orbits function
terms and speed up Monte Carlo runs.

Also has numerical ECI conversion to other basis sets, but has not been
strongly tested.
"""

__author__ = "Luis Barroso-Luque"

from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np
from monty.json import MSONable, jsanitize

from smol.cofe.space.clusterspace import ClusterSubspace


@dataclass
class RegressionData:
    """Dataclass used to store regression model details.

    This class is used to store the details used in fitting a cluster expansion
    for future reference and good provenance practices. It is highly
    recommended to initialize :class:`ClusterExpansion` objects with this
    class
    """

    module: str
    estimator_name: str
    feature_matrix: np.ndarray
    property_vector: np.ndarray
    parameters: dict

    @classmethod
    def from_object(cls, estimator, feature_matrix, property_vector, parameters=None):
        """Create a RegressionData object from an esimator class.

        Args:
            estimator (object):
                Estimator class or function.
            feature_matrix (ndarray):
                feature matrix used in fit.
            property_vector (ndarray):
                target property vector used in fit.
            parameters (dict):
                dictionary with pertinent fitting parameters.
                ie regularization, etc. It is highly recommended that you save
                this out of good practice and to ensure reproducibility.
        Returns:
            RegressionData
        """
        try:
            estimator_name = estimator.__class__.__name__
        except AttributeError:
            estimator_name = estimator.__name__

        return cls(
            module=estimator.__module__,
            estimator_name=estimator_name,
            feature_matrix=feature_matrix,
            property_vector=property_vector,
            parameters=parameters,
        )

    @classmethod
    def from_sklearn(cls, estimator, feature_matrix, property_vector):
        """Create a RegressionData object from sklearn estimator.

        Args:
            estimator (object):
                scikit-leanr estimator class or derived.
            feature_matrix (ndarray):
                feature matrix used in fit.
            property_vector (ndarray):
                target property vector used in fit.
        Returns:
            RegressionData
        """
        return cls(
            module=estimator.__module__,
            estimator_name=estimator.__class__.__name__,
            feature_matrix=feature_matrix,
            property_vector=property_vector,
            parameters=estimator.get_params(),
        )


class ClusterExpansion(MSONable):
    """Class for the ClusterExpansion proper.

    This needs a :class:`ClusterSubspace` and a corresponding set of
    coefficients from a fit.

    The main method to use this the predict method to predict the fitted
    property for new structures. This can be used to compare the accuracy
    of the fit with a set of test structures not used in training.

    Although this is purely optional and does not change the class performance,
    it is also recommended you save some information about learn metrics
    such as CV score, test/train rmse, or anything to quantify the "goodness"
    in the metadata dictionary. See for example learn metrics in
    :code:`sklearn.metrics` for many useful methods to get this quantities.

    This class is also used for Monte Carlo simulations to create a
    :class:`ClusterExpansionProcessor` that calculates the CE for a fixed
    supercell size.
    Before using a ClusterExpansion for Monte Carlo you should consider pruning
    the correlation/orbit functions with small coefficients or eci.

    Attributes:
        coefficients (ndarry): coefficients of the cluster expansion
        metadata (dict): dict to save optional values describing cluster
            expansion. i.e. if it was pruned, any error metrics etc.
    """

    def __init__(self, cluster_subspace, coefficients, regression_data=None):
        r"""Initialize a ClusterExpansion.

        Args:
            cluster_subspace (ClusterSubspace):
                A clustersubspace representing the subspace over which the
                Cluster Expansion was fit. Must be the same used to create
                the feature matrix.
            coefficients (ndarray):
                coefficients for cluster expansion. Make sure the supplied
                coefficients to the correlation vector terms (length and order)
                These correspond to the
                ECI x the multiplicity of orbit x multiplicity of bit ordering
            regression_data (RegressionData): optional
                RegressionData object with details used in the fit of the
                corresponding expansion. The feature_matrix attributed here is
                necessary to compute things like numerical ECI transormations
                for different bases.
        """
        if (
            regression_data is not None
            and len(coefficients) != regression_data.feature_matrix.shape[1]
        ):
            raise AttributeError(
                f"Feature matrix shape {regression_data.feature_matrix.shape} "
                f"does not match the number of coefficients "
                f"{len(coefficients)}."
            )

        if len(coefficients) != len(cluster_subspace):
            raise AttributeError(
                f"The size of the give subspace {len(cluster_subspace)} does "
                f"not match the number of coefficients {len(coefficients)}"
            )

        self.coefs = coefficients
        self.regression_data = regression_data
        self._subspace = cluster_subspace
        # make copy for possible changes/pruning
        self._feat_matrix = (
            regression_data.feature_matrix.copy()
            if regression_data is not None
            else None
        )
        self._eci = None

    @property
    def eci(self):
        """Get the eci for the cluster expansion.

        This just divides by the corresponding multiplicities. External terms
        will are dropped since their fitted coefficients do not represent ECI.
        """
        if self._eci is None:
            num_ext_terms = len(self._subspace.external_terms)  # check for extra terms
            coefs = self.coefs[:-num_ext_terms] if num_ext_terms else self.coefs[:]
            self._eci = coefs.copy()
            self._eci /= self._subspace.function_total_multiplicities
        return self._eci

    @property
    def structure(self):
        """Get primitive structure which the expansion is based on."""
        return self.cluster_subspace.structure

    @property
    def expansion_structure(self):
        """Get expansion structure.

        Prim structure with only sites included in the expansion.
        (i.e. sites with partial occupancies)
        """
        return self.cluster_subspace.expansion_structure

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

    @property
    def feature_matrix(self):
        """Get the feature matrix used in fit.

        If not given returns an identity matrix of len num_corrs
        """
        return (
            self._feat_matrix
            if self._feat_matrix is not None
            else np.eye(len(self.coefs))
        )

    def predict(self, structure, scmatrix=None, normalize=False):
        """Predict the fitted property for a given set of structures.

        Args:
            structure (Structure):
                Structures to predict from
            scmatrix (3*3 Arraylike): optional
                Supercell matrix of structure.
            normalize (bool): optional
                Whether to return the predicted property normalized
                by the prim cell size.
        Returns:
            float
        """
        corrs = self.cluster_subspace.corr_from_structure(
            structure, scmatrix=scmatrix, normalized=normalize
        )
        return np.dot(self.coefs, corrs)

    def prune(self, threshold=0, with_multiplicity=False):
        """Remove fit coefficients or ECI's with small values.

        Removes ECI's and and orbits in the ClusterSubspaces that have
        ECI/parameter values smaller than the given threshold.

        This will change the fits error metrics (ie RMSE) a little, but it
        should not be much. If they change a lot then the threshold used is
        probably too high and important functions are being pruned.

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
        bit_ids = [i for i, coef in enumerate(coefs) if abs(coef) < threshold]
        self.cluster_subspace.remove_corr_functions(bit_ids)
        # Update necessary attributes
        ids_complement = list(set(range(len(self.coefs))) - set(bit_ids))
        ids_complement.sort()
        self.coefs = self.coefs[ids_complement]
        if self._feat_matrix is not None:
            self._feat_matrix = self._feat_matrix[:, ids_complement]
        self._eci = None  # Reset

    def copy(self):
        """Return a copy of self."""
        return ClusterExpansion.from_dict(self.as_dict())

    def __str__(self):
        """Pretty string for printing."""
        corr = np.zeros(self.cluster_subspace.num_corr_functions)
        corr[0] = 1  # zero point cluster
        # This might need to be redefined to take "expectation" using measure
        feature_avg = np.average(self.feature_matrix, axis=0)
        feature_std = np.std(self.feature_matrix, axis=0)
        print_str = (
            "ClusterExpansion:\n    Prim Composition: "
            f"{self.structure.composition}\n"
            f"Num corr functions: {self.cluster_subspace.num_corr_functions}\n"
        )
        if self._feat_matrix is None:
            print_str += (
                "[Feature matrix used in fit was not provided. Feature "
                "statistics are meaningless.]\n"
            )
        ecis = (
            len(corr)
            * [
                0.0,
            ]
            if self.coefs is None
            else self.coefs
        )
        print_str += f"    [Orbit]  id: {str(0):<3}\n"
        print_str += "        bit       eci\n"
        print_str += f'        {"[X]":<10}{ecis[0]:<4.3}\n'
        for orbit in self.cluster_subspace.orbits:
            print_str += (
                f"    [Orbit]  id: {orbit.bit_id:<3} size: "
                f"{len(orbit.bits):<3} radius: "
                f"{orbit.base_cluster.diameter:<4.3}\n"
            )
            print_str += (
                "        id    bit       eci     feature avg  feature std  " "eci*std\n"
            )
            for i, bits in enumerate(orbit.bit_combos):
                eci = ecis[orbit.bit_id + i]
                f_avg = feature_avg[orbit.bit_id + i]
                f_std = feature_std[orbit.bit_id + i]
                print_str += (
                    f"        {orbit.bit_id + i:<6}{str(bits[0]):<10}"
                    f"{eci:<8.3f}{f_avg:<13.3f}{f_std:<13.3f}"
                    f"{eci*f_std:<.3f}\n"
                )
        return print_str

    @classmethod
    def from_dict(cls, d):
        """Create ClusterExpansion from serialized MSONable dict."""
        reg_data_dict = deepcopy(d.get("regression_data"))
        if reg_data_dict is not None:
            reg_data_dict["feature_matrix"] = np.array(
                d["regression_data"]["feature_matrix"]
            )
            reg_data_dict["property_vector"] = np.array(
                d["regression_data"]["property_vector"]
            )
            reg_data = RegressionData(**reg_data_dict)
        else:
            reg_data = None

        cluster_expansion = cls(
            ClusterSubspace.from_dict(d["cluster_subspace"]),
            coefficients=np.array(d["coefs"]),
            regression_data=reg_data,
        )

        # update copy of feature matrix to keep any changes
        if d["feature_matrix"] is not None:
            cls._feat_matrix = np.array(d["feature_matrix"])
        return cluster_expansion

    def as_dict(self):
        """
        Get Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        if self._feat_matrix is not None:
            feature_matrix = self._feat_matrix.tolist()
        else:
            feature_matrix = self._feat_matrix
        if self.regression_data is not None:
            reg_data = jsanitize(asdict(self.regression_data))
        else:
            reg_data = None

        ce_d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "cluster_subspace": self.cluster_subspace.as_dict(),
            "coefs": self.coefs.tolist(),
            "regression_data": reg_data,
            "feature_matrix": feature_matrix,
        }
        return ce_d
