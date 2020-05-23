"""
Implementation of a StructureWrangler and additional functions used to
preprocess and check (wrangle) fitting data of structures and properties.
Also functions to obtain weights by energy above hull or energy above
composition for a given set of structures.
"""

__author__ = "Luis Barroso-Luque"
__credits__ = "William Davidson Richard"

from collections import defaultdict
import numpy as np
from monty.json import MSONable
from pymatgen import Composition, Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from smol.cofe.extern import EwaldTerm
from smol.cofe.configspace.clusterspace import ClusterSubspace
from smol.exceptions import StructureMatchError
from smol.globals import kB


def weights_energy_above_composition(structures, energies, temperature=2000):
    """
    Computes weights for structure energy above the minimum reduced composition
    energy.
    Args:
        structures (list):
            list of pymatgen.Structures
        energies (array):
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
    """
    Computes weights for structure energy above the hull of all given
    structures

    Args:
        structures (list):
            list of pymatgen.Structures
        energies (array):
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
    """
    Class that handles (wrangles) input data structures and properties to fit
    in a cluster expansion. This class holds a ClusterSubspace used to compute
    correlation vectors and produce feature/design matrices used to fit the
    final ClusterExpansion.
    This class is meant to take all input training data in the form of
    (structure, properties) where the properties represent the target
    material property for the given structure that will be used to train
    the cluster expansion.
    The class takes care of returning the fitting data as a cluster
    correlation feature matrix. Weights for each structure can also be
    provided see the above functions to weight by energy above hull or
    energy above composition.
    This class also has methods to check/prepare/filter the data. A metadata
    dictionary is used to keep track of applied filters, but users can also use
    it to save any other pertinent information that will be saved with using
    as_dict for future reference.
    """

    def __init__(self, cluster_subspace):
        """
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
        return self._subspace

    @property
    def num_structures(self):
        return len(self._items)

    @property
    def structures(self):
        return [i['structure'] for i in self._items]

    @property
    def refined_structures(self):
        return [i['ref_structure'] for i in self._items]

    @property
    def supercell_matrices(self):
        return np.array([i['scmatrix'] for i in self._items])

    @property
    def structure_site_mappings(self):
        return [i['mapping'] for i in self._items]

    @property
    def feature_matrix(self):
        return np.array([i['features'] for i in self._items])

    @property
    def sizes(self):
        return np.array([i['size'] for i in self._items])

    @property
    def metadata(self):
        return self._metadata

    def add_properties(self, key, property_vector):
        """
        Add another property to structures already in the wrangler. The length
        of the property vector must match the number of structures contained,
        and should be in the same order.
        Args:
            key (str):
                name of property
            property_vector (array):
                array with the property for each structure
        """
        if self.num_structures != len(property_vector):
            raise AttributeError('Length of property_vector must match number'
                                 f'of structures {len(property_vector)} != '
                                 f'{self.num_structures}.')
        for prop, item in zip(property_vector, self._items):
            item['properties'][key] = prop

    def get_property_vector(self, key, normalize=False):
        """
        Gets the property target vector that can be used to fit the
        corresponding correlation feature matrix in a cluster expansion
        Args:
            key (str):
                name of the property
            normalize (bool): optional
                to normalize by prim size. If the property sought is not
                already normalized, you need to normalize before fitting a CE
        """
        properties = np.array([i['properties'][key] for i in self._items])
        if normalize:
            properties /= self.sizes

        return properties

    def get_weights(self, key):
        """
        Gets the weights specified by the given key
        Args:
            key (str):
                name of corresponding weights
        """
        return np.array([i['weights'][key] for i in self._items])

    def remove_structure(self, structure):
        """Will remove the corresponding structure and associated data."""
        try:
            index = self.structures.index(structure)
            del self._items[index]
        except ValueError:
            raise ValueError(f'Structure {structure} was not found. '
                             'Nothing has been removed.')

    def add_weights(self, key, weights):
        """
        Add weights to structures already in the wrangler. The length
        of the given weights must match the number of structures contained,
        and should be in the same order.
        Args:
            key (str):
                name describing weights
            weights (array):
                array with the weight for each structure
        """
        if self.num_structures != len(weights):
            raise AttributeError('Length of weights must match number of '
                                 f'structures {len(weights)} != '
                                 f'{self.num_structures}.')
        for weight, item in zip(weights, self._items):
            item['weights'][key] = weight

    def add_data(self, structure, properties, weights=None, verbose=False,
                 supercell_matrix=None, site_mapping=None):
        """
        Add a structure and measured property to Structure Wrangler.
        The property should ideally be extensive (i.e. not normalized per atom
        or unit cell, etc).
        Will attempt to computes correlation vector and if successful will
        add the structure otherwise it ignores that structure. Usually failures
        are caused by the Structure Matcher in the given ClusterSubspace.
        If using verbose, the errors will be printed out.

        Args
            structure (pymatgen.Structure):
                A fit structure
            properties (dict):
                A dictionary with a key describing the property and the target
                value for the corresponding structure. For example if only a
                single property {'energy': value} but can also add more than
                one i.e. {'total_energy': value1, 'formation_energy': value2}.
                You are free to make up the keys for each property but make
                sure you are consistent for all structures that you add.
            weights (dict):
                The weight given to the structure when doing the fit. The key
                must match at least one of the given properties.
            verbose (bool):
                if True then print structures that fail in StructureMatcher.
            supercell_matrix (array): optional
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
        """

        item = self._process_structure(structure, properties, weights, verbose,
                                       supercell_matrix, site_mapping)
        if item is not None:
            self._items.append(item)

    def change_subspace(self, cluster_subspace):
        """
        Will swap out the cluster subspace and update features accordingly.
        This is a faster operation than creating a new one. Can also be useful
        to create a copy and the change the subspace.

        Args:
            cluster_subspace:
                New subspace
        """
        self._subspace = cluster_subspace
        self.update_features()

    def update_features(self):
        """
        Update the features/feature matrix for the data held. This is useful
        when something is changed in the cluster_subspace after
        creating the Wrangler, for example added an Ewald term after creating
        the Wrangler.
        """
        for item in self._items:
            struct = item['structure']
            mat = item['scmatrix']
            map = item['mapping']
            item['features'] = self._subspace.corr_from_structure(struct,
                                                                  scmatrix=mat,
                                                                  site_mapping=map)  # noqa

    def remove_all_data(self):
        """Removes all data from Wrangler."""
        self._items = []

    def filter_by_ewald(self, max_ewald, verbose=False):
        """
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
        ewald_corr = None
        for term in self._subspace.external_terms:
            if isinstance(term, EwaldTerm):
                ewald_corr = [i['features'][-1] for i in self._items]
        if ewald_corr is None:
            ewald_corr = []
            for item in self._items:
                struct = item['structure']
                matrix = item['scmatrix']
                occu = self._subspace.occupancy_from_structure(struct,
                                                               encode=True,
                                                               scmatrix=matrix)
                supercell = self._subspace.structure.copy()
                supercell.make_supercell(matrix)
                size = self._subspace.num_prims_from_matrix(matrix)
                term = EwaldTerm().corr_from_occupancy(occu, supercell, size)
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

    @classmethod
    def from_dict(cls, d):
        """
        Creates Structure Wrangler from serialized MSONable dict.
        """
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
        """
        Json-serialization dict representation.

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

    def _process_structure(self, structure, properties, weights, verbose,
                           scmatrix, smapping):
        """
        Check if the structure for this data item can be matched to the cluster
        subspace prim structure to obtain its supercell matrix, correlation,
        and refined structure.
        """
        try:
            if scmatrix is None:
                scmatrix = self._subspace.scmatrix_from_structure(structure)
            size = self._subspace.num_prims_from_matrix(scmatrix)
            if smapping is None:
                supercell = self._subspace.structure.copy()
                supercell.make_supercell(scmatrix)
                smapping = self._subspace.structure_site_mapping(supercell,
                                                                 structure)
            fm_row = self._subspace.corr_from_structure(structure,
                                                        scmatrix=scmatrix,
                                                        site_mapping=smapping)  # noqa
            refined_struct = self._subspace.refine_structure(structure,
                                                             scmatrix)
            weights = {} if weights is None else weights
        except StructureMatchError as e:
            if verbose:
                print(f'Unable to match {structure.composition} with energy '
                      f'{property} to supercell_structure. Throwing out.\n'
                      f'Error Message: {str(e)}.')
            return
        return {'structure': structure, 'ref_structure': refined_struct,
                'properties': properties, 'weights': weights,
                'scmatrix': scmatrix, 'mapping': smapping,
                'features': fm_row, 'size': size}


def _energies_above_hull(structures, energies, ce_structure):
    """
    Computes energies above hull constructed from phase diagram of the
    given structures.
    """
    pd = _pd(structures, energies, ce_structure)
    e_above_hull = []
    for s, e in zip(structures, energies):
        entry = PDEntry(s.composition.element_composition, e)
        e_above_hull.append(pd.get_e_above_hull(entry))
    return np.array(e_above_hull)


def _pd(structures, energies, cs_structure):
    """
    Generate a phase diagram with the structures and energies.
    """
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
    """
    Computes structure energies above reduced composition.
    """

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
