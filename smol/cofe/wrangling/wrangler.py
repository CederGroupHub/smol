"""Implementation of a StructureWrangler.

A StructureWrangler is used to generate and organize training data to fit a
cluster expansion using the terms defined in a ClusterSubspace. It takes care
of computing the training features (correlations) to construct a feature matrix
to be used along with a target property vector to obtain the coefficients for
a cluster expansion using some linear regression model.

Includes functions used to preprocess and check (wrangling) fitting data of
structures and properties.
"""

__author__ = "Luis Barroso-Luque"
__credits__ = "William Davidson Richard"

import warnings
from importlib import import_module
from itertools import combinations
from typing import Sequence

import numpy as np
from monty.json import MSONable, jsanitize
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry

from smol.utils.exceptions import StructureMatchError


class StructureWrangler(MSONable):
    """Class to create fitting data to fit a cluster expansion.

    A StructureWrangler handles (wrangles) input data structures and properties
    to fit in a cluster expansion. This class holds a ClusterSubspace used to
    compute correlation vectors and produce feature/design matrices used to fit
    the final ClusterExpansion.

    This class is meant to take all input training data in the form of
    (structure, properties) where the properties represent the target
    material property for the given structure that will be used to train
    the cluster expansion, and returns the fitting data as a cluster
    correlation feature matrix (orbit basis function values).

    This class also has methods to check/prepare/filter the data. A metadata
    dictionary is used to keep track of applied filters, but users can also use
    it to save any other pertinent information that will be saved with using
    :code:`StructureWrangler.as_dict` for future reference. Other preprocessing
    and filtering methods are available in tools.py and select.py.
    """

    def __init__(self, cluster_subspace):
        """Initialize a StructureWrangler.

        Args:
            cluster_subspace (ClusterSubspace):
                a ClusterSubspace object that will be used to fit a
                ClusterExpansion with the provided data.
        """
        self._subspace = cluster_subspace
        self._entries = []
        self._ind_sets = {}  # data indices for test/training splits etc
        self._metadata = {"applied_filters": []}

    @property
    def cluster_subspace(self):
        """Get the underlying ClusterSubspace used to compute features."""
        return self._subspace

    @property
    def num_structures(self):
        """Get number of structures added (correctly matched to prim)."""
        return len(self._entries)

    @property
    def num_features(self):
        """Get number of features for each added structure."""
        return self.feature_matrix.shape[1]

    @property
    def available_properties(self):
        """Get list of properties that have been added."""
        return list(
            {p for entry in self._entries for p in entry.data["properties"].keys()}
        )

    @property
    def available_indices(self):
        """Get list of available data index sets."""
        return list(self._ind_sets.keys())

    @property
    def available_weights(self):
        """Get list of weights that have been added."""
        return list(
            {p for entry in self._entries for p in entry.data["weights"].keys()}
        )

    @property
    def structures(self):
        """Get list of included structures."""
        return [entry.structure for entry in self._entries]

    @property
    def refined_structures(self):
        """Get list of refined structures."""
        return [entry.data["refined_structure"] for entry in self._entries]

    @property
    def feature_matrix(self):
        """Get feature matrix.

        Rows are structures, and columns are correlation vectors.
        """
        return np.array([entry.data["correlations"] for entry in self._entries])

    @property
    def sizes(self):
        """Get sizes of each structure in terms of number of prims."""
        return np.array([entry.data["size"] for entry in self._entries])

    @property
    def occupancy_strings(self):
        """Get occupancy strings for each structure in the StructureWrangler."""
        occupancies = [
            self._subspace.occupancy_from_structure(
                entry.structure,
                entry.data["supercell_matrix"],
                entry.data["site_mapping"],
            )
            for entry in self._entries
        ]
        return occupancies

    @property
    def supercell_matrices(self):
        """Get list of supercell matrices relating each structure to prim."""
        return np.array([entry.data["supercell_matrix"] for entry in self._entries])

    @property
    def structure_site_mappings(self):
        """Get list of site mappings for each structure to prim."""
        return [entry.data["site_mapping"] for entry in self._entries]

    @property
    def entries(self):
        """Get a list of the entry dictionaries."""
        return self._entries

    def data_indices(self, key):
        """Get a specific data index set."""
        return self._ind_sets[key]

    @property
    def metadata(self):
        """Get dictionary to save applied filters, etc."""
        return self._metadata

    def get_feature_matrix_rank(self, rows=None, cols=None):
        """Get the rank of the feature matrix or a submatrix of it.

        Args:
            rows (list):
                indices of structures to include in feature matrix.
            cols (list):
                indices of features (correlations) to include in feature matrix

        Returns:
            int: the rank of the matrix
        """
        rows = rows if rows is not None else range(self.num_structures)
        cols = cols if cols is not None else range(self.num_features)
        return np.linalg.matrix_rank(self.feature_matrix[rows][:, cols])

    def get_feature_matrix_orbit_rank(self, orbit_id, rows=None):
        """Get the rank of an orbit submatrix of the feature matrix.

        Args:
            orbit_id (int):
                Orbit id to obtain sub feature matrix rank of.
            rows (list): optional
                list of row indices corresponding to structures to include.

        Returns:
            int: rank of orbit sub feature matrix
        """
        columns = [
            i
            for i, oid in enumerate(self._subspace.function_orbit_ids)
            if oid == orbit_id
        ]
        return self.get_feature_matrix_rank(rows=rows, cols=columns)

    def get_condition_number(self, rows=None, cols=None, norm_p=2):
        """Compute the condition number for the feature matrix or submatrix.

        The condition number is a measure of how sensitive the solution to
        the linear system is to perturbations in the sampled data. The larger
        the condition number the more ill-conditioned the linear problem is.

        Args:
            rows (list):
                indices of structures to include in feature matrix.
            cols (list):
                indices of features (correlations) to include in feature matrix
            norm_p : (optional)
                the type of norm to use when computing condition number.
                See the numpy docs for np.linalg.cond for options.

        Returns:
            float: matrix condition number
        """
        rows = rows if rows is not None else range(self.num_structures)
        cols = cols if cols is not None else range(self.num_features)
        return np.linalg.cond(self.feature_matrix[rows][:, cols], p=norm_p)

    def get_gram_matrix(self, rows=None, cols=None, normalize=True):
        r"""Compute the Gram matrix for the feature matrix or a submatrix.

        The Gram matrix, :math:`G = X^TX`, where each entry is
        :math:`G_{ij} = X_i \cdot X_j`. By default, G will have each column
        (feature vector) normalized. This makes it possible to compare Gram
        matrices for different feature matrix size or using different basis
        sets. This ensures every entry satisfies :math:`-1 \le G_{ij} \le 1`.

        Args:
            rows (list):
                indices of structures to include in feature matrix.
            cols (list):
                indices of features (correlations) to include in feature matrix
            normalize:
                if True (default), will normalize each feature vector in the
                feature matrix.
        Returns:
            ndarray: Gram matrix
        """
        rows = rows if rows is not None else range(self.num_structures)
        cols = cols if cols is not None else range(self.num_features)
        matrix = self.feature_matrix[rows][:, cols]
        if normalize:
            matrix /= np.sqrt(matrix.T.dot(matrix).diagonal())
        return matrix.T.dot(matrix)

    def get_duplicate_corr_indices(
        self, cutoffs=None, decimals=12, rm_external_terms=True
    ):
        """Find indices of rows with duplicate corr vectors in feature matrix.

        Args:
            cutoffs (dict): optional
                dictionary with cluster diameter cutoffs for correlation
                functions to consider in correlation vectors.
            decimals (int): optional
                number of decimals to round correlations in order to allow
                some numerical tolerance for finding duplicates. If None,
                no rounding will be done. Beware that using a ClusterSubspace
                with an orthogonal site basis will likely be off by some
                numerical tolerance so rounding is recommended.
            rm_external_terms (bool) : optional
                if True, will not consider external terms and only consider
                correlations proper when looking for duplicates.
        Returns:
            list: list containing lists of indices of rows in feature_matrix
            where duplicates occur
        """
        if len(self.feature_matrix) == 0:
            duplicate_inds = []
        else:
            num_ext = len(self.cluster_subspace.external_terms)
            if cutoffs is not None:
                inds = self._subspace.function_inds_from_cutoffs(cutoffs)
            else:
                inds = range(self.num_features - num_ext)
            if not rm_external_terms:
                inds.extend(range(-1, -num_ext - 1, -1))

            feature_matrix = (
                self.feature_matrix
                if decimals is None
                else np.around(
                    self.feature_matrix, decimals, self.feature_matrix.copy()
                )
            )
            _, inverse = np.unique(feature_matrix[:, inds], return_inverse=True, axis=0)
            duplicate_inds = [
                list(np.where(inverse == i)[0])
                for i in np.unique(inverse)
                if len(np.where(inverse == i)[0]) > 1
            ]
        return duplicate_inds

    def get_matching_corr_duplicate_indices(
        self, decimals=12, structure_matcher=None, **matcher_kwargs
    ):
        """Find indices of equivalent structures.

        Args:
            decimals (int): optional
                number of decimals to round correlations in order to allow
                some numerical tolerance for finding duplicates.
            structure_matcher (StructureMatcher): optional
                StructureMatcher object to use for matching structures.
            **matcher_kwargs:
                keyword arguments to use when initializing a StructureMatcher
                if not given

        Returns:
            list: list of lists of equivalent structures (that match) and have
            duplicate correlation vectors.
        """
        matcher = (
            structure_matcher
            if structure_matcher is not None
            else StructureMatcher(**matcher_kwargs)
        )
        duplicate_indices = self.get_duplicate_corr_indices(decimals)

        matching_inds = []
        for inds in duplicate_indices:
            # match all combinations of duplicates
            matches = [
                set(c)
                for c in combinations(inds, 2)
                if matcher.fit(
                    self.structures[c[0]], self.structures[c[1]], symmetric=True
                )
            ]

            while overlaps := list(
                filter(lambda s: s[0] & s[1], combinations(matches, 2))
            ):
                all_overlaps = [o for overlap in overlaps for o in overlap]
                # keep only disjoint sets
                matches = [s for s in matches if s not in all_overlaps]
                # add union of overlapping sets
                for set1, set2 in overlaps:
                    if set1 | set2 not in matches:
                        matches.append(set1 | set2)
                overlaps = list(filter(lambda s: s[0] & s[1], combinations(matches, 2)))
            matching_inds += [list(sorted(m)) for m in matches]
        return matching_inds

    def get_constant_features(self):
        """Find indices of constant feature vectors (columns).

        A constant feature vector means the corresponding correlation function
        evaluates to the exact same value for all included structures, meaning
        it does not really help much when fitting. Many constant feature
        vectors may be a sign of insufficient sampling of configuration space.

        Excludes the empty cluster, which is by definition constant.

        Returns:
            ndarray: array of column indices.
        """
        arr = self.feature_matrix
        col_mask = np.all(arr == arr[0, :], axis=0)
        return np.where(col_mask == 1)[0][1:]

    def get_similarity_matrix(self, rows=None, cols=None, rtol=1e-5):
        """Get similarity matrix of correlation vectors.

        Generate a matrix to compare the similarity of correlation feature
        vectors (columns) in the feature matrix. Matrix element a(i,j)
        represents the fraction of equivalent corresponding values in feature
        vectors i and j.
        This construction is analogous to the Gram matrix, but instead of an
        inner product, it counts the number of identical corresponding elements
        in feature vectors i and j.

        Args:
            rows (list):
                indices of structures to include in feature matrix.
            cols (list):
                indices of features (correlations) to include in feature matrix
            rtol (float):
                relative tolerance for comparing feature matrix column values

        Returns:
            ndarray:
                (n x n) similarity matrix
        """
        rows = rows if rows is not None else range(self.num_structures)
        cols = cols if cols is not None else range(self.num_features)

        matrix = self.feature_matrix[rows][:, cols]
        num_structs = len(rows)
        num_corrs = len(cols)
        sim_matrix = np.identity(num_corrs)
        for i in range(num_corrs):
            for j in range(i + 1, num_corrs):
                num_identical = np.sum(
                    np.isclose(matrix[:, i], matrix[:, j], rtol=rtol)
                )
                sim_matrix[i, j] = num_identical / num_structs
                sim_matrix[j, i] = num_identical / num_structs

        return sim_matrix

    def get_property_vector(self, key, normalize=True):
        """Get the property target vector.

        The property target vector to be used to fit the corresponding
        correlation feature matrix to obtain coefficients for a cluster
        expansion. It should always be properly/consistently normalized when
        used for a fit.

        Args:
            key (str):
                name of the property
            normalize (bool): optional
                if True, normalizes by prim size. If the property sought is not
                already normalized, you need to normalize before fitting a CE.
        """
        if key in self.available_properties:
            properties = np.array(
                [entry.data["properties"][key] for entry in self._entries]
            )
        else:
            try:
                properties = np.array([getattr(entry, key) for entry in self._entries])
            except AttributeError as error:
                raise ValueError(
                    f"{key} is not a valid property. The property must be a valid "
                    f"ComputedStructureEntry property or one of "
                    f"{self.available_properties}"
                ) from error

        if normalize:
            properties = properties / self.sizes

        return properties

    def add_data_indices(self, key, indices):
        """Add a set of data indices.

        For example, use this for saving test/training splits or separating
        duplicates.
        """
        if not isinstance(indices, (Sequence, np.ndarray)):
            raise TypeError("indices must be Sequence like or an ndarray.")

        if any(i not in range(self.num_structures) for i in indices):
            raise ValueError("One or more indices are out of range.")
        self._ind_sets[key] = list(indices)

    def get_weights(self, key):
        """Get the weights specified by the given key.

        Args:
            key (str):
                name of corresponding weights
        """
        return np.array([entry.data["weights"][key] for entry in self._entries])

    def add_entry(
        self,
        entry,
        properties=None,
        weights=None,
        supercell_matrix=None,
        site_mapping=None,
        verbose=True,
        raise_failed=False,
    ):
        """Add a structure and measured property to the StructureWrangler.

        The energy and properties need to be extensive (i.e. not normalized per atom
        or unit cell, directly from DFT).

        An attempt to compute the correlation vector is made and if successful the
        structure is successfully added. Otherwise the structure is ignored.
        Usually failures are caused by the StructureMatcher in the given
        ClusterSubspace failing to map structures to the primitive structure.

        Args:
            entry (ComputedStructureEntry):
                A ComputedStructureEntry with a training structure, energy and
                properties
            properties (dict):
                Dictionary with a key describing the property and the target
                value for the corresponding structure. For example if only a
                single property {'energy': value} but can also add more than
                one, i.e. {'total_energy': value1, 'formation_energy': value2}.
                You are free to make up the keys for each property but make
                sure you are consistent for all structures that you add.
            weights (dict):
                the weight given to the structure when doing the fit. The key
                must match at least one of the given properties.
            supercell_matrix (ndarray): optional
                if the corresponding structure has already been matched to the
                ClusterSubspace prim structure, passing the supercell_matrix
                will use that instead of trying to re-match. If using this,
                the user is responsible for having the correct supercell_matrix.
                Here you are the cause of your own bugs.
            site_mapping (list): optional
                site mapping as obtained by StructureMatcher.get_mapping
                such that the elements of site_mapping represent the indices
                of the matching sites to the prim structure. If you pass this
                option, you are fully responsible that the mappings are correct!
            verbose (bool): optional
                if True, will raise warning regarding  structures that fail in
                StructureMatcher, and structures that have duplicate corr vectors.
            raise_failed (bool): optional
                if True, will raise the thrown error when adding a structure
                that  fails. This can be helpful to keep a list of structures that
                fail for further inspection.
        """
        processed_entry = self.process_entry(
            entry,
            properties,
            weights,
            supercell_matrix,
            site_mapping,
            verbose,
            raise_failed,
        )
        if processed_entry is not None:
            self._entries.append(processed_entry)
        if verbose:
            self._corr_duplicate_warning(self.num_structures - 1)

    def append_entries(self, entries):
        """Append a list of entries.

        Each entry must have all necessary fields. A entry can be
        obtained using the process_structure method.

        Args:
            entries (list of ComputedStructureEntry):
                list of entries with all necessary information
        """
        required_keys = [
            "refined_structure",
            "supercell_matrix",
            "site_mapping",
            "correlations",
            "size",
        ]

        for i, entry in enumerate(entries):
            if not all(key in entry.data.keys() for key in required_keys):
                raise ValueError(
                    f"entry {i} is missing required keys. Make sure"
                    " they were obtained with the process_structure method."
                )
            if len(self._entries) > 0:
                if set(entry.data["properties"].keys()) != set(
                    self.available_properties
                ):
                    raise ValueError(
                        f"The properties in the entry being added do not match all "
                        f"the properties that have already been added: "
                        f"{self.available_properties}.\n"
                        f" Additional entries must include "
                        f"the same properties included."
                    )

                if set(entry.data["weights"].keys()) != set(self.available_weights):
                    raise ValueError(
                        f"The properties in the entry being added do not match all "
                        f"the weights that have already been added: "
                        f"{self.available_weights}.\n"
                        f" Additional entries must include the"
                        f" same weights included."
                    )
            self._entries.append(entry)
            self._corr_duplicate_warning(self.num_structures - 1)

    def add_weights(self, key, weights):
        """Add weights to structures already in the StructureWrangler.

        The length of the given weights must match the number of structures
        contained, and should be in the same order.

        Args:
            key (str):
                name describing weights
            weights (ndarray):
                array with the weight for each structure
        """
        if self.num_structures != len(weights):
            raise AttributeError(
                "Length of weights must match number of structures "
                f"{len(weights)} != {self.num_structures}."
            )
        for weight, entry in zip(weights, self._entries):
            entry.data["weights"][key] = weight

    def add_properties(self, key, property_vector):
        """Add another property vector to structures already in the StructureWrangler.

        The length of the property vector must match the number of structures
        contained, and should be in the same order so that the property
        corresponds to the correct structure.

        Args:
            key (str):
                name of property
            property_vector (ndarray):
                array with the property for each structure
        """
        if self.num_structures != len(property_vector):
            raise AttributeError(
                "Length of property_vector must match number of structures"
                f" {len(property_vector)} != {self.num_structures}."
            )

        for prop, entry in zip(property_vector, self._entries):
            entry.data["properties"][key] = prop

    def remove_properties(self, *property_keys):
        """Remove properties with given keys.

        Args:
            *property_keys (str):
                names of properties to remove
        """
        for key in property_keys:
            try:
                for entry in self._entries:
                    del entry.data["properties"][key]
            except KeyError:
                warnings.warn(f"Property {key} does not exist.", RuntimeWarning)

    def remove_entry(self, entry):
        """Remove a given structure and associated data."""
        try:
            index = self._entries.index(entry)
            del self._entries[index]
        except ValueError:
            warnings.warn(
                f"Entry {entry} was not found. Nothing has been removed.",
                RuntimeWarning,
            )

    def change_subspace(self, cluster_subspace):
        """Change the underlying ClusterSubspace.

        Will swap out the ClusterSubspace and update features accordingly.
        This is a faster operation than creating a new one. Can also be useful
        to create a copy and then change the subspace.

        Args:
            cluster_subspace:
                New ClusterSubspace to be used for determining features.
        """
        self._subspace = cluster_subspace
        self.update_features()

    def update_features(self):
        """Update the features/feature matrix for the data held.

        This is useful when something is changed in the ClusterSubspace after
        creating the StructureWrangler, for example when adding an Ewald term
        after the StructureWrangler has already been created. This will prevent
        having to re-match structures and such.
        """
        for entry in self._entries:
            structure = entry.structure
            scmatrix = entry.data["supercell_matrix"]
            mapping = entry.data["site_mapping"]
            entry.data["correlations"] = self._subspace.corr_from_structure(
                structure, scmatrix=scmatrix, site_mapping=mapping
            )

    def remove_all_data(self):
        """Remove all data from the StructureWrangler."""
        self._entries = []

    def process_entry(
        self,
        entry,
        properties=None,
        weights=None,
        supercell_matrix=None,
        site_mapping=None,
        verbose=False,
        raise_failed=False,
    ):
        """Process a ComputedStructureEntry to be added to StructureWrangler.

        Checks if the structure for this entry can be matched to the
        ClusterSubspace prim structure to obtain its supercell matrix,
        correlation, and refined structure. If so, the entry will be updated by adding
        these to its data dictionary.

        Args:
            entry (ComputedStructureEntry):
                A ComputedStructureEntry corresponding to a training structure and
                properties
            properties (dict): optional
                A dictionary with a keys describing the property and the target
                value for the corresponding structure. Energy and corrected energy
                should already be in the ComputedStructureEntry so there is no need
                to pass it here.
                You are free to make up the keys for each property but make
                sure you are consistent for all structures that you add.
            weights (dict): optional
                The weight given to the structure when doing the fit. The key
                must match at least one of the given properties.
            supercell_matrix (ndarray): optional
                if the corresponding structure has already been matched to the
                ClusterSubspace prim structure, passing the supercell_matrix
                will use that instead of trying to re-match. If using this
                the user is responsible to have the correct supercell_matrix.
                Here you are the cause of your own bugs.
            site_mapping (list): optional
                site mapping as obtained by
                :code:`StructureMatcher.get_mapping`
                such that the elements of site_mapping represent the indices
                of the matching sites to the prim structure. If you pass this
                option, you are fully responsible that the mappings are correct!
            verbose (bool):
                if True, will raise warning for structures that fail in
                StructureMatcher, and structures that have duplicate corr vectors.
            raise_failed (bool): optional
                if True, will raise the thrown error when adding a structure
                that fails. This can be helpful to keep a list of structures that
                fail for further inspection.

        Returns:
            ComputedStructureEntry: entry with CE pertinent properties
        """
        if len(self._entries) > 0:
            if properties is not None and len(self.available_properties) > 0:
                if set(properties.keys()) != set(self.available_properties):
                    print(set(properties.keys()))
                    raise ValueError(
                        f"The properties entry being added do not match all "
                        f"the properties that have already been added: "
                        f"{self.available_properties}.\n Additional entries must "
                        f"include the same properties included."
                    )

            if weights is not None and len(self.available_properties) > 0:
                if set(weights.keys()) != set(self.available_weights):
                    raise ValueError(
                        f"The properties in the entry being added do not match all the"
                        f"weights that have already been added: "
                        f"{self.available_weights}.\n Additional entries must include "
                        f"the same weights included."
                    )

        properties = {} if properties is None else properties
        weights = {} if weights is None else weights

        try:
            if supercell_matrix is None:
                supercell_matrix = self._subspace.scmatrix_from_structure(
                    entry.structure
                )

            size = self._subspace.num_prims_from_matrix(supercell_matrix)
            if site_mapping is None:
                supercell = self._subspace.structure.copy()
                supercell.make_supercell(supercell_matrix)
                site_mapping = self._subspace.structure_site_mapping(
                    supercell, entry.structure
                )

            fm_row = self._subspace.corr_from_structure(
                entry.structure, scmatrix=supercell_matrix, site_mapping=site_mapping
            )
            refined_struct = self._subspace.refine_structure(
                entry.structure, scmatrix=supercell_matrix, site_mapping=site_mapping
            )

        except StructureMatchError as s_match_error:
            if verbose:
                warnings.warn(
                    f"Unable to match {entry.composition} with "
                    f"energy {entry.energy} to supercell_structure. "
                    f"Throwing out.\n Error Message: {str(s_match_error)}",
                    UserWarning,
                )
            if raise_failed:
                raise s_match_error
            return None

        entry.data["refined_structure"] = refined_struct
        entry.data["supercell_matrix"] = supercell_matrix
        entry.data["site_mapping"] = site_mapping
        entry.data["correlations"] = fm_row
        entry.data["size"] = size
        property_dict = entry.data.get("properties")
        if property_dict is not None:
            property_dict.update(properties)
        else:
            entry.data["properties"] = properties

        weight_dict = entry.data.get("weights")
        if weight_dict is not None:
            weight_dict.update(weights)
        else:
            entry.data["weights"] = weights

        return entry

    def _corr_duplicate_warning(self, index):
        """Warn if corr vector of entry with given index is duplicated."""
        for duplicate_inds in self.get_duplicate_corr_indices():
            if index in duplicate_inds:
                duplicates = "".join(
                    f"Index {i} - {self._entries[i].structure.composition} "
                    f"energy={self._entries[i].energy}\n"
                    for i in duplicate_inds
                )
                warnings.warn(
                    "The following structures have duplicated correlation "
                    f"vectors:\n {duplicates} Consider adding more terms to "
                    "the clustersubspace or filtering duplicates.",
                    UserWarning,
                )

    @classmethod
    def from_dict(cls, d, energy_key=None):
        """Create Structure Wrangler from an MSONable dict.

        Args:
            d (dict):
                MSON dict of StructureWrangler
            energy_key (str): optional
                energy property key, for legacy files

        Returns:
            StructureWrangler
        """
        module = import_module(d["_subspace"]["@module"])
        subspace_cls = getattr(module, d["_subspace"]["@class"])
        wrangler = cls(cluster_subspace=subspace_cls.from_dict(d["_subspace"]))

        # check for legacy files
        if "_items" in d.keys():
            warnings.warn(
                "This StructureWrangler was created with a previous version of smol.\n"
                "Please re-save it to prevent this warning.",
                FutureWarning,
            )
            if energy_key is None:
                raise RuntimeWarning(
                    "Please load using from_dict method and provide the key used for "
                    "the energy property to correctly load this StructureWrangler."
                )
            entries = []
            for item in d["_items"]:
                energy = item["properties"].pop(energy_key)
                data = {
                    "properties": item["properties"],
                    "weights": item["weights"],
                    "supercell_matrix": np.array(item["scmatrix"]),
                    "correlations": np.array(item["features"]),
                    "size": item["size"],
                    "site_mapping": item["mapping"],
                    "refined_structure": Structure.from_dict(item["ref_structure"]),
                }
                entries.append(
                    ComputedStructureEntry(
                        Structure.from_dict(item["structure"]), energy, data=data
                    )
                )
        else:
            entries = [
                ComputedStructureEntry.from_dict(entry_d) for entry_d in d["_entries"]
            ]

        wrangler._entries = entries
        wrangler._metadata = d["metadata"]
        wrangler._ind_sets = d.get("_ind_sets") or {}
        return wrangler

    def as_dict(self):
        """Get Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        wrangler_dict = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "_subspace": self._subspace.as_dict(),
            "_entries": jsanitize(self._entries, strict=True),
            "_ind_sets": jsanitize(self._ind_sets, strict=True),  # jic for int's
            "metadata": self.metadata,
        }
        return wrangler_dict
