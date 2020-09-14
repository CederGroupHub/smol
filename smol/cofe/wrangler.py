"""Implementation of a StructureWrangler and functions to obtain fit weights.

A StructureWrangler is used to generate and organize training data to fit a
cluster expansion using the terms defined in a ClusterSubpace. It takes care
of computing the training features (correlations) to construct a feature matrix
to be used along with a target property vector to obtain the coefficients for
a cluster expansion using some linear regression model.

Includes functions used to preprocess and check (wrangle) fitting data of
structures and properties.

Also functions to obtain weights by energy above hull or energy above
composition for a given set of structures.
"""

__author__ = "Luis Barroso-Luque"
__credits__ = "William Davidson Richard"

import warnings
from collections import defaultdict
import numpy as np
from monty.json import MSONable
from pymatgen import Composition, Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from smol.cofe.extern import EwaldTerm
from smol.cofe.space.clusterspace import ClusterSubspace
from smol.exceptions import StructureMatchError
from smol.constants import kB


def weights_energy_above_composition(structures, energies, temperature=2000):
    """Compute weights for energy above the minimum reduced composition energy.

    Args:
        structures (list):
            list of pymatgen.Structures
        energies (ndarray):
            energies of corresponding structures.
        temperature (float):
            temperature to used in boltzmann weight

    Returns: weights for each structure.
        array
    """
    e_above_comp = _energies_above_composition(structures, energies)
    return np.exp(-e_above_comp / (kB * temperature))


def weights_energy_above_hull(structures, energies, cs_structure,
                              temperature=2000):
    """Compute weights for structure energy above the hull of given structures.

    Args:
        structures (list):
            list of pymatgen.Structures
        energies (ndarray):
            energies of corresponding structures.
        cs_structure (Structure):
            The pymatgen.Structure used to define the cluster subspace
        temperature (float):
            temperature to used in boltzmann weight.

    Returns: weights for each structure.
        array
    """
    e_above_hull = _energies_above_hull(structures, energies, cs_structure)
    return np.exp(-e_above_hull / (kB * temperature))


class StructureWrangler(MSONable):
    """Class to create fitting data to fit a cluster expansion.

    A StructureWrangler handles (wrangles) input data structures and properties
    to fit in a cluster expansion. This class holds a ClusterSubspace used to
    compute correlation vectors and produce feature/design matrices used to fit
    the final ClusterExpansion.

    This class is meant to take all input training data in the form of
    (structure, properties) where the properties represent the target
    material property for the given structure that will be used to train
    the cluster expansion.

    The class takes care of returning the fitting data as a cluster
    correlation feature matrix (orbit basis function values). Weights for each
    structure can also be provided see the above functions to weight by energy
    above hull or energy above composition.

    This class also has methods to check/prepare/filter the data. A metadata
    dictionary is used to keep track of applied filters, but users can also use
    it to save any other pertinent information that will be saved with using
    :code:`StructureWrangler.as_dict` for future reference.
    """

    def __init__(self, cluster_subspace):
        """Initialize a StructureWrangler.

        Args:
            cluster_subspace (ClusterSubspace):
                A ClusterSubspace object that will be used to fit a
                ClusterExpansion with the provided data.
        """
        self._subspace = cluster_subspace
        self._items = []
        self._metadata = {'applied_filters': []}

    @property
    def cluster_subspace(self):
        """Get the underlying ClusterSubspace used to compute features."""
        return self._subspace

    @property
    def num_structures(self):
        """Get number of structures added (correctly matched to prim)."""
        return len(self._items)

    @property
    def num_features(self):
        """Get umber of features for each added structure."""
        return self.feature_matrix.shape[1]

    @property
    def available_properties(self):
        """Get list of properties that have been added."""
        return list(set(p for i in self._items
                        for p in i['properties'].keys()))

    @property
    def available_weights(self):
        """Get list of weights that have been added."""
        return list(set(p for i in self._items
                        for p in i['weights'].keys()))

    @property
    def structures(self):
        """Get list of included structures."""
        return [i['structure'] for i in self._items]

    @property
    def refined_structures(self):
        """Get list of refined structures."""
        return [i['ref_structure'] for i in self._items]

    @property
    def feature_matrix(self):
        """Get feature matrix.

        Rows are structures, Columns are correlations.
        """
        return np.array([i['features'] for i in self._items])

    @property
    def sizes(self):
        """Get sizes of each structure in terms of number of prims."""
        return np.array([i['size'] for i in self._items])

    @property
    def occupancy_strings(self):
        """Get occupancy strings for each of the structures in the wrangler."""
        occupancies = [self._subspace.occupancy_from_structure(i['structure'],
                                                               i['scmatrix'],
                                                               i['mapping'])
                       for i in self._items]
        return occupancies

    @property
    def supercell_matrices(self):
        """Get list of supercell matrices relating each structure to prim."""
        return np.array([i['scmatrix'] for i in self._items])

    @property
    def structure_site_mappings(self):
        """Get list of site mappings for each structure to prim."""
        return [i['mapping'] for i in self._items]

    @property
    def data_items(self):
        """Get a list of the data item dictionaries."""
        return self._items

    @property
    def metadata(self):
        """Get dictionary to save applied filters, etc."""
        return self._metadata

    def get_matrix_rank(self, rows=None, cols=None):
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

    def get_condition_number(self, rows=None, cols=None, p=2):
        """Compute the condition number for the feature matrix or submatrix.

        The condition number is a measure of how sensitive the solution to
        the linear system is to perturbations in the sampled data. The larger
        the condition number the more ill-conditioned the linear problem is.

        Args:
            rows (list):
                indices of structures to include in feature matrix.
            cols (list):
                indices of features (correlations) to include in feature matrix
            p : (optional)
                the type of norm to use when computing condition number.
                see the numpy docs for np.linalg.cond for options.

        Returns:
            float: matrix condition number
        """
        rows = rows if rows is not None else range(self.num_structures)
        cols = cols if cols is not None else range(self.num_features)
        return np.linalg.cond(self.feature_matrix[rows][:, cols], p=p)

    def add_data(self, structure, properties, normalized=False, weights=None,
                 verbose=False, supercell_matrix=None, site_mapping=None,
                 raise_failed=False):
        """Add a structure and measured property to the StructureWrangler.

        The properties are usually extensive (i.e. not normalized per atom
        or unit cell, directly from DFT). If the properties have already been
        normalized then set normalized to True. Users need to make sure their
        normalization is consistent. (Default normalization is per the
        primititive structure of the given cluster subspace)

        An attempt to computes correlation vector is made and if successful the
        structure is succesfully added otherwise it ignores that structure.
        Usually failures are caused by the Structure Matcher in the given
        ClusterSubspace failing to match structures to the primitive structure.

        Args:
            structure (Structure):
                A fit structure
            properties (dict):
                A dictionary with a key describing the property and the target
                value for the corresponding structure. For example if only a
                single property {'energy': value} but can also add more than
                one i.e. {'total_energy': value1, 'formation_energy': value2}.
                You are free to make up the keys for each property but make
                sure you are consistent for all structures that you add.
            normalized (bool):
                Whether the given properties have already been normalized.
            weights (dict):
                The weight given to the structure when doing the fit. The key
                must match at least one of the given properties.
            verbose (bool):
                if True then print structures that fail in StructureMatcher.
            supercell_matrix (ndarray): optional
                If the corresponding structure has already been matched to the
                clustersubspace prim structure, passing the supercell_matrix
                will use that instead of trying to re-match. If using this
                the user is responsible to have the correct supercell_matrix,
                Here you are the cause of your own bugs.
            site_mapping (list): optional
                Site mapping as obtained by StructureMatcher.get_mapping
                such that the elements of site_mapping represent the indices
                of the matching sites to the prim structure. I you pass this
                option you are fully responsible that the mappings are correct!
            raise_failed (bool): optional
                If true will raise the thrown error when adding a structure
                fails. This can be helpful to keep a list of structures that
                fail for further inspection.
        """
        item = self.process_structure(structure, properties, normalized,
                                      weights, verbose, supercell_matrix,
                                      site_mapping, raise_failed)
        if item is not None:
            self._items.append(item)
        for duplicate_inds in self.get_duplicate_corr_inds():
            if self.num_structures - 1 in duplicate_inds:
                duplicates = [f"{self._items[i]['structure'].composition} "
                              f"{self._items[i]['properties']}\n"
                              for i in duplicate_inds]
                warnings.warn("The following structures have duplicated " 
                              f"correlation vectors:\n{duplicates}"
                              "Consider adding more terms to the"
                              "clustersubspace or filtering duplicates.",
                              UserWarning)

    def append_data_items(self, data_items):
        """Append a list of data items.

        Each data item must have all necessary fields. A data item can be
        obtained using the process_structure method.

        Args:
            data_items (list of dict):
                list of data items with all necessary information
        """
        keys = ['structure', 'ref_structure', 'properties', 'weights',
                'scmatrix', 'mapping', 'features', 'size']
        for item in data_items:
            if not all(key in keys for key in item.keys()):
                raise ValueError('A supplied data item is missing required '
                                 'keys. Make sure they were obtained with '
                                 'the process_structure method.')
            self._items.append(item)

    def add_weights(self, key, weights):
        """Add weights to structures already in the wrangler.

        The length of the given weights must match the number of structures
        contained, and should be in the same order.

        Args:
            key (str):
                Name describing weights
            weights (ndarray):
                Array with the weight for each structure
        """
        if self.num_structures != len(weights):
            raise AttributeError('Length of weights must match number of '
                                 f'structures {len(weights)} != '
                                 f'{self.num_structures}.')
        for weight, item in zip(weights, self._items):
            item['weights'][key] = weight

    def add_properties(self, key, property_vector, normalized=False):
        """Add another property vector to structures already in the wrangler.

        The length of the property vector must match the number of structures
        contained, and should be in the same order such that the property
        corresponds to the correct structure.

        Args:
            key (str):
                Name of property
            property_vector (ndarray):
                Array with the property for each structure
            normalized (bool): (optional)
                Wether the given properties have already been normalized.
        """
        if self.num_structures != len(property_vector):
            raise AttributeError('Length of property_vector must match number'
                                 f'of structures {len(property_vector)} != '
                                 f'{self.num_structures}.')
        if normalized:
            property_vector *= self.sizes

        for prop, item in zip(property_vector, self._items):
            item['properties'][key] = prop

    def remove_properties(self, *property_keys):
        """Remove properties from given keys.

        Args:
            *property_keys (str):
                names of properties to remove
        """
        for key in property_keys:
            try:
                for item in self._items:
                    del item['properties'][key]
            except KeyError:
                warnings.warn(f'Propertiy {key} does not exist.',
                              RuntimeWarning)

    def get_property_vector(self, key, normalize=True):
        """Get the property target vector.

        The property targent vector that be used to fit the corresponding
        correlation feature matrix to obtain coefficients for a cluster
        expansion. It should always be properly/consistently normalized when
        used for a fit.

        Args:
            key (str):
                Name of the property
            normalize (bool): optional
                To normalize by prim size. If the property sought is not
                already normalized, you need to normalize before fitting a CE.
        """
        properties = np.array([i['properties'][key] for i in self._items])
        if normalize:
            properties /= self.sizes

        return properties

    def get_weights(self, key):
        """Get the weights specified by the given key.

        Args:
            key (str):
                Name of corresponding weights
        """
        return np.array([i['weights'][key] for i in self._items])

    def remove_structure(self, structure):
        """Remove a given structure and associated data."""
        try:
            index = self.structures.index(structure)
            del self._items[index]
        except ValueError:
            raise ValueError(f'Structure {structure} was not found. '
                             'Nothing has been removed.')

    def change_subspace(self, cluster_subspace):
        """Change the underlying cluster subspace.

        Will swap out the cluster subspace and update features accordingly.
        This is a faster operation than creating a new one. Can also be useful
        to create a copy and the change the subspace.

        Args:
            cluster_subspace:
                New subspace to be used for determining features.
        """
        self._subspace = cluster_subspace
        self.update_features()

    def update_features(self):
        """Update the features/feature matrix for the data held.

        This is useful when something is changed in the cluster_subspace after
        creating the Wrangler, for example added an Ewald term after creating
        the Wrangler. This will prevent having to match structures and such.
        """
        for item in self._items:
            struct = item['structure']
            mat = item['scmatrix']
            mapp = item['mapping']
            item['features'] = self._subspace.corr_from_structure(struct,
                                                                  scmatrix=mat,
                                                                  site_mapping=mapp)  # noqa

    def remove_all_data(self):
        """Remove all data from Wrangler."""
        self._items = []

    def process_structure(self, structure, properties, normalized=False,
                          weights=None, verbose=False, supercell_matrix=None,
                          site_mapping=None, raise_failed=False):
        """Process a structure to be added to wrangler.

        Checks if the structure for this data item can be matched to the
        cluster subspace prim structure to obtain its supercell matrix,
        correlation, and refined structure.

        Args:
            structure (Structure):
                A structure corresponding to the given properties
            properties (dict):
                A dictionary with a key describing the property and the target
                value for the corresponding structure. For example if only a
                single property {'energy': value} but can also add more than
                one i.e. {'total_energy': value1, 'formation_energy': value2}.
                You are free to make up the keys for each property but make
                sure you are consistent for all structures that you add.
            normalized (bool):
                Wether the given properties have already been normalized.
            weights (dict):
                The weight given to the structure when doing the fit. The key
                must match at least one of the given properties.
            verbose (bool):
                if True then print structures that fail in StructureMatcher.
            supercell_matrix (ndarray): optional
                If the corresponding structure has already been matched to the
                clustersubspace prim structure, passing the supercell_matrix
                will use that instead of trying to re-match. If using this
                the user is responsible to have the correct supercell_matrix,
                Here you are the cause of your own bugs.
            site_mapping (list): optional
                Site mapping as obtained by
                :code:`StructureMatcher.get_mapping`
                such that the elements of site_mapping represent the indices
                of the matching sites to the prim structure. If you pass this
                option you are fully responsible that the mappings are correct!
            raise_failed (bool): optional
                If true will raise the thrown error when adding a structure
                fails. This can be helpful to keep a list of structures that
                fail for further inspection.

        Returns:
            dict: data item dict for structure
        """
        try:
            if supercell_matrix is None:
                supercell_matrix = self._subspace.scmatrix_from_structure(structure)  # noqa
            size = self._subspace.num_prims_from_matrix(supercell_matrix)
            if site_mapping is None:
                supercell = self._subspace.structure.copy()
                supercell.make_supercell(supercell_matrix)
                site_mapping = self._subspace.structure_site_mapping(supercell,
                                                                     structure)
            fm_row = self._subspace.corr_from_structure(structure,
                                                        scmatrix=supercell_matrix,  # noqa
                                                        site_mapping=site_mapping)  # noqa
            refined_struct = self._subspace.refine_structure(structure,
                                                             supercell_matrix)
            if normalized:
                properties = {key: val*size for key, val in properties.items()}
            weights = {} if weights is None else weights
        except StructureMatchError as e:
            if verbose:
                print(f'Unable to match {structure.composition} with '
                      f'properties {properties} to supercell_structure. '
                      f'Throwing out.\n Error Message: {str(e)}')
            if raise_failed:
                raise e
            return
        return {'structure': structure, 'ref_structure': refined_struct,
                'properties': properties, 'weights': weights,
                'scmatrix': supercell_matrix, 'mapping': site_mapping,
                'features': fm_row, 'size': size}

    def filter_by_ewald(self, max_ewald, verbose=False):
        """Filter structures by electrostatic interaction energy.

        Filter the input structures to only use those with low electrostatic
        energies (no large charge separation in cell). This energy is
        referenced to the lowest value at that composition. Note that this is
        before the division by the relative dielectric constant and is per
        primitive cell in the cluster expansion -- 1.5 eV/atom seems to be a
        reasonable value for dielectric constants around 10.

        Args:
            max_ewald (float):
                Ewald threshold. The maximum Ewald energy, normalized by prim
        """
        warnings.warn('the filter_by_ewald method is going to be deprecated.\n'
                      'The functionality will still be available but with '
                      'a different general filters interface.',
                      category=DeprecationWarning, stacklevel=2)
        ewald_corr = None
        for term in self._subspace.external_terms:
            if isinstance(term, EwaldTerm):
                ewald_corr = [i['features'][-1] for i in self._items]
        if ewald_corr is None:
            ewald_corr = []
            for item in self._items:
                struct = item['structure']
                matrix = item['scmatrix']
                mapp = item['mapping']
                occu = self._subspace.occupancy_from_structure(struct,
                                                               encode=True,
                                                               scmatrix=matrix,
                                                               site_mapping=mapp)  # noqa
                supercell = self._subspace.structure.copy()
                supercell.make_supercell(matrix)
                size = self._subspace.num_prims_from_matrix(matrix)
                term = EwaldTerm().value_from_occupancy(occu, supercell)/size
                ewald_corr.append(term)

        min_e = defaultdict(lambda: np.inf)
        for corr, item in zip(ewald_corr, self._items):
            c = item['structure'].composition.reduced_composition
            if corr < min_e[c]:
                min_e[c] = corr

        items = []
        for corr, item in zip(ewald_corr, self._items):
            mine = min_e[item['structure'].composition.reduced_composition]
            r_e = corr - mine
            if r_e > max_ewald:
                if verbose:
                    print(f'Skipping {item["structure"].composition} '
                          f'Ewald interaction energy is {r_e}.')
            else:
                items.append(item)
        removed = len(self._items) - len(items)
        metadata = {'Ewald': {'max_ewald': max_ewald,
                              'nstructs_removed': removed,
                              'nstructs_total': len(self._items)}}
        self._metadata['applied_filters'].append(metadata)
        self._items = items

    def get_duplicate_corr_inds(self):
        """Find indices of rows with duplicate corr vectors in feature matrix.

        Returns:
            list: list containing lists of indices of rows in feature_matrix
            where duplicates occur
        """
        if len(self.feature_matrix) == 0:
            duplicate_inds = []
        else:
            num_ext = len(self.cluster_subspace.external_terms)
            end = self.feature_matrix.shape[1] - num_ext - 1
            _, inverse = np.unique(self.feature_matrix[:, :end],
                                   return_inverse=True, axis=0)
            duplicate_inds = [list(np.where(inverse == i)[0])
                              for i in np.unique(inverse)
                              if len(np.where(inverse == i)[0]) > 1]
        return duplicate_inds

    def filter_duplicate_corrs(self, key, filter_by='min'):
        """Filter structures with duplicate correlation.

        Keep structures with the min or max value of the given property.

        Args:
            key (str):
                Name of property to consider when returning structure indices
                for the feature matrix with duplicate corr vectors removed.
            filter_by (str)
                The criteria for the property value to keep. Options are min
                or max
        """
        warnings.warn('filter_duplicate_corrs method is going to be '
                      'deprecated.\n The functionality will still be available'
                      'with a different general filters interface.',
                      category=DeprecationWarning, stacklevel=2)
        if filter_by not in ('max', 'min'):
            raise ValueError(f'Filtering by {filter_by} is not an option.')

        choose_val = np.argmin if filter_by == 'min' else np.argmax

        property_vector = self.get_property_vector(key)
        dupe_inds = self.get_duplicate_corr_inds()
        to_keep = {inds[choose_val(property_vector[inds])]
                   for inds in dupe_inds}
        to_remove = set(i for inds in dupe_inds for i in inds) - to_keep
        self._items = [item for i, item in enumerate(self._items)
                       if i not in to_remove]

    @classmethod
    def from_dict(cls, d):
        """Create Structure Wrangler from an MSONable dict."""
        sw = cls(cluster_subspace=ClusterSubspace.from_dict(d['_subspace']))
        items = []
        for item in d['_items']:
            items.append({'properties': item['properties'],
                          'structure':
                              Structure.from_dict(item['structure']),
                          'ref_structure':
                              Structure.from_dict(item['ref_structure']),
                          'scmatrix': np.array(item['scmatrix']),
                          'mapping': item['mapping'],
                          'features': np.array(item['features']),
                          'size': item['size'],
                          'weights': item['weights']})
        sw._items = items
        sw._metadata = d['metadata']
        return sw

    def as_dict(self):
        """Get Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        s_items = []
        for item in self._items:
            s_items.append({'properties': item['properties'],
                            'structure': item['structure'].as_dict(),
                            'ref_structure':
                                item['ref_structure'].as_dict(),
                            'scmatrix': item['scmatrix'].tolist(),
                            'features': item['features'].tolist(),
                            'mapping': item['mapping'],
                            'size': item['size'],
                            'weights': item['weights']})
        d = {'@module': self.__class__.__module__,
             '@class': self.__class__.__name__,
             '_subspace': self._subspace.as_dict(),
             '_items': s_items,
             'metadata': self.metadata}
        return d


def _energies_above_hull(structures, energies, ce_structure):
    """Compute energies above hull.

    Hull is constructed from phase diagram of the given structures.
    """
    pd = _pd(structures, energies, ce_structure)
    e_above_hull = []
    for s, e in zip(structures, energies):
        entry = PDEntry(s.composition.element_composition, e)
        e_above_hull.append(pd.get_e_above_hull(entry))
    return np.array(e_above_hull)


def _pd(structures, energies, cs_structure):
    """Generate a phase diagram with the structures and energies."""
    entries = []

    for s, e in zip(structures, energies):
        entries.append(PDEntry(s.composition.element_composition, e))

    max_e = max(entries, key=lambda e: e.energy_per_atom).energy_per_atom
    max_e += 1000
    for el in cs_structure.composition.keys():
        entry = PDEntry(Composition({el: 1}).element_composition, max_e)
        entries.append(entry)

    return PhaseDiagram(entries)


def _energies_above_composition(structures, energies):
    """Compute structure energies above reduced composition."""
    min_e = defaultdict(lambda: np.inf)
    for s, e in zip(structures, energies):
        comp = s.composition.reduced_composition
        if e / len(s) < min_e[comp]:
            min_e[comp] = e / len(s)
    e_above = []
    for s, e in zip(structures, energies):
        comp = s.composition.reduced_composition
        e_above.append(e / len(s) - min_e[comp])
    return np.array(e_above)
