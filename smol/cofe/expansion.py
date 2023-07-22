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
from functools import cached_property

import numpy as np
from monty.json import MSONable, jsanitize

from smol.cofe.space.clusterspace import ClusterSubspace
from smol.utils.cluster import get_orbit_data


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
        """Create a RegressionData object from an estimator class.

        Args:
            estimator (object):
                Estimator class or function.
            feature_matrix (ndarray):
                feature matrix used in fit.
            property_vector (ndarray):
                target property vector used in fit.
            parameters (dict):
                Dictionary with pertinent fitting parameters,
                i.e. regularization, etc. It is highly recommended that you save
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
                scikit-learn estimator class or derived class.
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

    The main method to use is the :meth:`predict` method to predict the fitted
    property for new structures. This can be used to compare the accuracy
    of the fit with a set of test structures not used in training.

    Although this is purely optional and does not change the class performance,
    it is also recommended you save some information about learn metrics
    such as CV score, test/train rmse, or anything to quantify the "goodness"
    in the metadata dictionary. See for example learn metrics in
    :code:`sklearn.metrics` for many useful methods to get these quantities.

    This class is also used for Monte Carlo simulations to create a
    :class:`ClusterExpansionProcessor` that calculates the CE for a fixed
    supercell size.
    Before using a ClusterExpansion for Monte Carlo you should consider pruning
    the correlation/orbit functions with very small coefficients or eci.

    Attributes:
        coefficients (ndarry): coefficients of the ClusterExpansion
        metadata (dict): dict to save optional values describing cluster
            expansion. i.e. if it was pruned, any error metrics etc.
    """

    def __init__(self, cluster_subspace, coefficients, regression_data=None):
        r"""Initialize a ClusterExpansion.

        Args:
            cluster_subspace (ClusterSubspace):
                a ClusterSubspace representing the subspace over which the
                ClusterExpansion was fit. Must be the same used to create
                the feature matrix.
            coefficients (ndarray):
                coefficients for cluster expansion. Make sure the supplied
                coefficients match the correlation vector terms (length and order)
                These correspond to the
                ECI x the multiplicity of orbit x multiplicity of bit ordering.
            regression_data (RegressionData): optional
                RegressionData object with details used in the fit of the
                corresponding expansion. The feature_matrix attribute here is
                necessary to compute things like numerical ECI transformations
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
        self._subspace = cluster_subspace.copy()
        self._set_evaluator_data()

        # make copy for possible changes/pruning
        self._feat_matrix = (
            regression_data.feature_matrix.copy()
            if regression_data is not None
            else None
        )

    @cached_property
    def eci(self):
        """Get the ECI for the cluster expansion.

        This just divides coefficients by the corresponding multiplicities.
        External terms are dropped since their fitted coefficients do not
        represent ECI.
        """
        num_ext_terms = len(self._subspace.external_terms)  # check for extra terms
        coefs = self.coefs[:-num_ext_terms] if num_ext_terms else self.coefs[:]
        eci = coefs.copy()
        eci = eci / self._subspace.function_total_multiplicities
        return eci

    @cached_property
    def cluster_interaction_tensors(self):
        """Get tuple of cluster interaction tensors.

        Tuple of ndarrays where each array is the interaction tensor for the
        corresponding orbit of clusters.
        """
        interaction_tensors = (self.coefs[0],) + tuple(
            sum(
                m * self.eci[orbit.bit_id + i] * tensor
                for i, (m, tensor) in enumerate(
                    zip(orbit.bit_combo_multiplicities, orbit.correlation_tensors)
                )
            )
            for orbit in self._subspace.orbits
        )
        return interaction_tensors

    @property
    def structure(self):
        """Get primitive structure which the expansion is based on."""
        return self.cluster_subspace.structure

    @property
    def expansion_structure(self):
        """Get expansion structure.

        Prim structure with only sites included in the expansion
        (i.e. sites with partial occupancies)
        """
        return self.cluster_subspace.expansion_structure

    @property
    def cluster_subspace(self):
        """Get ClusterSubspace."""
        return self._subspace

    @property
    def eci_orbit_ids(self):
        """Get Orbit ids corresponding to each ECI in the Cluster Expansion.

        If the Cluster Expansion includes external terms these are not included
        in the list since they are not associated with any orbit.
        """
        return self._subspace.function_orbit_ids

    @property
    def effective_cluster_weights(self):
        """Calculate the cluster weights.

        The cluster weights are defined as the weighted sum of ECI squared, where
        the weights are the ordering multiplicities.
        """
        weights = np.array(
            [
                np.sum(
                    self._subspace.function_ordering_multiplicities[
                        self._subspace.function_orbit_ids == i
                    ]
                    * self.eci[self.eci_orbit_ids == i] ** 2
                )
                for i in range(len(self._subspace.orbits) + 1)
            ]
        )
        return weights

    @property
    def feature_matrix(self):
        """Get the feature matrix used in fit.

        If not given, returns an identity matrix of len num_corrs
        """
        return self._feat_matrix

    def predict(self, structure, normalized=False, scmatrix=None, site_mapping=None):
        """Predict the fitted property for a given set of structures.

        Args:
            structure (Structure):
                Structures to predict from
            normalized (bool): optional
                Whether to return the predicted property normalized
                by the prim cell size.
            scmatrix (ndarray): optional
                supercell matrix relating the prim structure to the given
                structure. Passing this if it has already been matched will
                make things much quicker. You are responsible that the
                supercell matrix is correct.
            site_mapping (list): optional
                Site mapping as obtained by
                :code:`StructureMatcher.get_mapping` such that the elements of
                site_mapping represent the indices of the matching sites to the prim
                structure. If you pass this option, you are fully responsible that the
                mappings are correct!
        Returns:
            float
        """
        corrs = self.cluster_subspace.corr_from_structure(
            structure,
            scmatrix=scmatrix,
            normalized=normalized,
            site_mapping=site_mapping,
        )
        return np.dot(self.coefs, corrs)

    def cluster_interactions_from_structure(
        self, structure, normalized=True, scmatrix=None, site_mapping=None
    ):
        """Compute the vector of cluster interaction values for given structure.

        A cluster interaction is simply a vector made up of the sum of all cluster
        expansion terms over the same orbit.

        Args:
            structure (Structure):
                Structures to predict from
            normalized (bool):
                Whether to return the predicted property normalized by
                the prim cell size.
            scmatrix (ndarray): optional
                supercell matrix relating the prim structure to the given
                structure. Passing this if it has already been matched will
                make things much quicker. You are responsible that the
                supercell matrix is correct.
            site_mapping (list): optional
                Site mapping as obtained by
                :code:`StructureMatcher.get_mapping` such that the elements of
                site_mapping represent the indices of the matching sites to the prim
                structure. If you pass this option, you are fully responsible that the
                mappings are correct!

        Returns: ndarray
            vector of cluster interaction values
        """
        if scmatrix is None:
            scmatrix = self._subspace.scmatrix_from_structure(structure)

        occu = self.cluster_subspace.occupancy_from_structure(
            structure, scmatrix=scmatrix, site_mapping=site_mapping, encode=True
        )
        indices = self._subspace.get_orbit_indices(scmatrix)
        interactions = self._subspace.evaluator.interactions_from_occupancy(
            occu, indices.container
        )

        if not normalized:
            interactions *= self._subspace.num_prims_from_matrix(scmatrix)

        return interactions

    def prune(self, threshold=0, with_multiplicity=False):
        """Remove fit coefficients or ECI's with small values.

        Removes ECI's and orbits in the ClusterSubspaces that have
        ECI/parameter values smaller than the given threshold.

        This will change the fits error metrics (i.e. RMSE) a little, but it
        should not be much. If they change a lot then the threshold used is
        probably too high and important functions are being pruned.

        This will not re-fit the ClusterExpansion. Note that if you re-fit
        after pruning, the ECI will probably change and hence also the fit
        performance.

        Args:
            threshold (float):
                threshold below which to remove.
            with_multiplicity (bool):
                if True, threshold is applied to the ECI proper, otherwise to
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

        if hasattr(self, "eci"):  # reset cache
            del self.eci

        if hasattr(self, "cluster_interaction_tensors"):  # reset cache
            del self.cluster_interaction_tensors

        # reset the evaluator
        self._set_evaluator_data(set_orbits=True)

    def copy(self):
        """Return a copy of self."""
        return ClusterExpansion.from_dict(self.as_dict())

    def _set_evaluator_data(self, set_orbits=False):
        """Set the orbit and cluster interaction data in evaluator."""
        if set_orbits:
            self._subspace.evaluator.reset_data(
                get_orbit_data(self._subspace.orbits),
                self._subspace.num_orbits,
                self._subspace.num_corr_functions,
            )

        flat_interaction_tensors = tuple(
            np.ravel(tensor, order="C")
            for tensor in self.cluster_interaction_tensors[1:]
        )
        self._subspace.evaluator.set_cluster_interactions(
            flat_interaction_tensors, offset=self.cluster_interaction_tensors[0]
        )

    def __str__(self):
        """Pretty string for printing."""
        outs = str(self.cluster_subspace).split("\n")[:6]

        if self.regression_data is not None:
            # This might need to be redefined to take "expectation" using measure
            feature_avg = np.average(self.feature_matrix, axis=0)
            feature_std = np.std(self.feature_matrix, axis=0)
            outs += [
                f"Regression Data : estimator={self.regression_data.estimator_name}",
                f"                  module={self.regression_data.module}",
                f"                  parameters={self.regression_data.parameters}",
                f"Target Property    : "
                f"mean={np.mean(self.regression_data.property_vector):0.4f}  "
                f"std={np.std(self.regression_data.property_vector):0.4f}",
            ]
        fit_var = sum(
            self._subspace.function_total_multiplicities[1:] * self.eci[1:] ** 2
        )
        outs += [
            f"ECI-based Property : mean={self.eci[0]:0.4f}"
            f"  std={np.sqrt(fit_var):0.4f}",
            "Fit Summary",
        ]

        for i, term in enumerate(self._subspace.external_terms):
            outs.append(f"{repr(term)}={self.coefs[len(self.eci) + i]:0.3f}")

        if self.regression_data is not None:
            outs += [
                " ---------------------------------------------------------------------"
                "-------------------------------",
                " |  ID    Orbit ID    Degree    Cluster Diameter    ECI    Feature AVG"
                "    Feature STD    ECI * STD  |",
                f" |  0        0          0              NA         "
                f"{self.eci[0]:^7.3f}{feature_avg[0]:^15.3f}"
                f"{feature_std[0]:^15.3f}{feature_std[0] * self.eci[0]:^13.3f}|",
            ]
        else:
            outs += [
                " ---------------------------------------------------------",
                " |  ID    Orbit ID    Degree    Cluster Diameter    ECI  |",
                f" |   0       0          0              NA        "
                f"{self.eci[0]:^7.3f} |",
            ]

        for degree, orbits in self.cluster_subspace.orbits_by_size.items():
            for orbit in orbits:
                for i, bits in enumerate(orbit.bit_combos):
                    line = (
                        f" |{orbit.bit_id + i:^6}{orbit.id:^12}{degree:^10}"
                        f"{orbit.base_cluster.diameter:^20.4f}"
                        f"{self.eci[orbit.bit_id + i]:^7.3f}"
                    )
                    if self.regression_data is not None:
                        line += (
                            f"{feature_avg[orbit.bit_id + i]:^15.3f}"
                            f"{feature_std[orbit.bit_id + i]:^15.3f}"
                            f"{feature_std[orbit.bit_id + i] * self.eci[orbit.bit_id + i]:^13.3f}"  # noqa
                        )
                    line += "|"
                    outs.append(line)
        outs.append(" " + (len(outs[-1]) - 1) * "-")
        return "\n".join(outs)

    def __repr__(self):
        """Return summary of expansion."""
        outs = ["Cluster Expansion Summary"]
        outs += repr(self.cluster_subspace).split("\n")[1:]

        if self.regression_data is not None:
            outs += [
                f"Regression Data : estimator={self.regression_data.estimator_name}"
                f"  module={self.regression_data.module}",
                f"  parameters={self.regression_data.parameters}",
                f"Target Property    : "
                f"mean={np.mean(self.regression_data.property_vector):0.4f}  "
                f"std={np.std(self.regression_data.property_vector):0.4f}",
            ]
        fit_var = sum(
            self._subspace.function_total_multiplicities[1:] * self.eci[1:] ** 2
        )
        outs += [
            f"ECI-based Property : mean={self.eci[0]:0.4f}  std={np.sqrt(fit_var):0.4f}"
        ]
        return "\n".join(outs)

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
