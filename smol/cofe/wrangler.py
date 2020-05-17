"""
Implementation of a StructureWrangler and additional functions used to
preprocess and check (wrangle) fitting data of structures and properties.
"""

from __future__ import division
from collections import defaultdict
import warnings
from types import MethodType
from functools import partial
import numpy as np
from monty.json import MSONable
from pymatgen import Composition, Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from smol.cofe.configspace import EwaldTerm
from smol.cofe.configspace.clusterspace import ClusterSubspace
from smol.exceptions import StructureMatchError
from smol.globals import kB


def weights_e_above_comp(wrangler, temperature=2000):
    """
    Computes weights for structure energy above the minimum reduced composition
    energy.

    Args:
        wrangler (StructureWrangler):
            A StructureWrangler to obtain structures and energies
        temperature (float):
            temperature to used in boltzmann weight

    Returns: weights for each structure.
        array
    """
    e_above_comp = _energies_above_composition(wrangler.structures,
                                               wrangler.properties)
    return np.exp(-e_above_comp / (kB * temperature))


def weights_e_above_hull(wrangler, temperature=2000):
    """
    Computes weights for structure energy above the hull of all given
    structures

    Args:
        wrangler (StructureWrangler):
            A StructureWrangler to obtain structures and energies
        temperature (float):
            temperature to used in boltzmann weight

    Returns: weights for each structure.
        array
    """
    e_above_hull = _energies_above_hull(wrangler.structures,
                                        wrangler.properties,
                                        wrangler.cluster_subspace.structure)
    return np.exp(-e_above_hull / (kB * temperature))


def _energies_above_composition(structures, energies):
    """
    Computes structure energies above reduced composition
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


def _energies_above_hull(structures, energies, ce_structure):
    """
    Computes energies above hull constructed from phase diagram of the
    given structures
    """
    pd = _pd(structures, energies, ce_structure)
    e_above_hull = []
    for s, e in zip(structures, energies):
        entry = PDEntry(s.composition.element_composition, e)
        e_above_hull.append(pd.get_e_above_hull(entry))
    return np.array(e_above_hull)


def _pd(structures, energies, cs_structure):
    """
    Generate a phase diagram with the structures and energies
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


# TODO should allow a bunch of properties saved by keys...
class StructureWrangler(MSONable):
    """
    Class that handles (wrangles) input data structures and properties to fit
    in a cluster expansion. This class holds a ClusterSubspace used to compute
    correlation vectors and produce feature/design matrices used to fit the
    final ClusterExpansion.
    """
    weight_funs = {'composition': weights_e_above_comp,
                   'hull': weights_e_above_hull}

    def __init__(self, cluster_subspace, weights=None, **weight_kwargs):
        """
        This class is meant to take all input training data in the form of
        (structure, property) where the property is usually (lets be honest
        always) the energy for the given structure.
        The class takes care of returning the fitting data as a cluster
        correlation design matrix.
        This class also has methods to check/prepare/filter the data, hence
        the name Wrangler from datawrangler.

        Args:
            cluster_subspace (ClusterSubspace):
                A ClusterSubspace object that will be used to fit a
                ClusterExpansion with the provided data.
            weights (str):
                str specifying type of weights (current: 'hull', 'composition')
            weight_kwargs:
                key word arguments for the type of weights specified
        """
        self._subspace = cluster_subspace

        if weights is not None:
            if weights not in self.weight_funs.keys():
                raise ValueError(f'{weights} is not an available weight '
                                 ' function. Choose one of '
                                 f'{list(self.weight_funs.keys())}.')
            weight_partial = partial(self.weight_funs[weights], **weight_kwargs)
            self.get_weights = MethodType(weight_partial, self)

        self._items = []
        self._metadata = {'weights': weights, 'applied_filters': []}

    @property
    def cluster_subspace(self):
        return self._subspace

    @property
    def items(self):
        return self._items

    @property
    def structures(self):
        return [i['structure'] for i in self._items]

    @property
    def refined_structures(self):
        return [i['refined_structure'] for i in self._items]

    @property
    def properties(self):
        return np.array([i['property'] for i in self._items])

    @property
    def normalized_properties(self):
        return self.properties / self.sizes

    @property
    def supercell_matrices(self):
        return np.array([i['scmatrix'] for i in self._items])

    @property
    def feature_matrix(self):
        return np.array([i['features'] for i in self._items])

    @property
    def sizes(self):
        return np.array([i['size'] for i in self._items])

    @property
    def weights(self):
        weights = [item.get('weight') for item in self._items
                   if item.get('weight') is not None]
        if len(weights) == len(self._items):
            return np.array(weights)
        elif self._metadata['weights'] is not None:
            # (re)compute weights if data was added
            weights = self.get_weights()
            for item, weight in zip(self._items, weights):
                item['weight'] = weight
            return weights

    @property
    def metadata(self):
        return self._metadata

    def add_data(self, structure, property, weight=None, verbose=False):
        """
        Add a structure and measured property to Structure Wrangler.
        The property should be extensive (i.e. not normalized per atom or unit
        cell, etc)
        Will attempt to computes correlation vector and if successful will
        add the structure otherwise it ignores that structure. Usually failures
        are caused by the Structure Matcher. If using verbose, the errors
        will be printed out.

        Args
            structure (pymatgen.Structure):
                A fit structure
            property (float):
                Fit property for the given structure. (usually energy)
            weight (float):
                the weight given to the structure when doing the fit. Only use
                this if weight_type was not specified when instantiating. If a
                weight_type was given when instantiating the class, the
                provided weight will be ignored.
            verbose (bool):
                if True then print structures that fail in StructureMatcher
        """

        item = self._process_structure(structure, property, verbose)

        if item is not None:
            if weight is not None:
                if self._metadata['weights'] is None:
                    item['weight'] = weight
                else:
                    warnings.warn(f"Weight type: {self.metadata['weights']} is"
                                  ' already specified. The provided weight '
                                  'will be ignored.')
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
            struct, mat = item['structure'], item['scmatrix']
            item['features'] = self._subspace.corr_from_structure(struct, mat)

    def remove_all_data(self):
        """Removes all data from Wrangler"""
        self._items = []

    def _process_structure(self, structure, property, verbose):
        """
        Check if the structure for this data item can be matched to the cluster
        subspace prim structure to obtain its supercell matrix, correlation,
        and refined structure.
        """
        try:
            scmatrix = self._subspace.scmatrix_from_structure(structure)
            size = self._subspace.num_prims_from_matrix(scmatrix)
            fm_row = self._subspace.corr_from_structure(structure, scmatrix)
            refined_struct = self._subspace.refine_structure(structure,
                                                             scmatrix)
        except StructureMatchError as e:
            if verbose:
                print(f'Unable to match {structure.composition} with energy '
                      f'{property} to supercell_structure. Throwing out.\n'
                      f'Error Message: {str(e)}.')
            return
        return {'structure': structure, 'refined_structure': refined_struct,
                'property': property, 'scmatrix': scmatrix, 'features': fm_row,
                'size': size}

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
                Ewald threshold
        """
        ewald_corr = None
        for term, args, kwargs in self._subspace.external_terms:
            if term.__name__ == 'EwaldTerm':
                if 'use_inv_r' in kwargs.keys() and kwargs['use_inv_r']:
                    raise NotImplementedError('cant use inv_r with max_ewald')
                ewald_corr = [i['features'][-1] for i in self._items]
        if ewald_corr is None:
            ewald_corr = []
            for struct in self.structures:
                scmatrix = self._subspace.scmatrix_from_structure(struct)
                occu = self._subspace.occupancy_from_structure(struct,
                                                               scmatrix,
                                                               encode=True)
                supercell = self._subspace.structure.copy()
                supercell.make_supercell(scmatrix)
                orb_inds = self._subspace.supercell_orbit_mappings(scmatrix)
                term = EwaldTerm.corr_from_occu(occu, supercell, orb_inds)
                ewald_corr.append(term)

        min_e = defaultdict(lambda: np.inf)
        for ecorr, item in zip(ewald_corr, self._items):
            c = item['structure'].composition.reduced_composition
            if ecorr < min_e[c]:
                min_e[c] = ecorr

        items = []
        for ecorr, item in zip(ewald_corr, self._items):
            mine = min_e[item['structure'].composition.reduced_composition]
            r_e = ecorr - mine
            if r_e > max_ewald:
                if verbose:
                    print(f'Skipping {item["structure"].composition} '
                          f'ewald energy is {r_e}')
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
        Creates Structure Wrangler from serialized MSONable dict
        """
        sw = cls(cluster_subspace=ClusterSubspace.from_dict(d['_subspace']))
        items = []
        for item in d['items']:
            items.append({'property': item['property'],
                          'structure':
                              Structure.from_dict(item['structure']),
                          'refined_structure':
                              Structure.from_dict(item['refined_structure']),
                          'scmatrix': np.array(item['scmatrix']),
                          'features': np.array(item['features']),
                          'size': item['size']})
        sw._items = items
        sw._metadata = d['metadata']
        return sw

    def as_dict(self):
        """
        Json-serialization dict representation

        Returns:
            MSONable dict
        """
        s_items = []
        for item in self._items:
            s_items.append({'property': item['property'],
                            'structure': item['structure'].as_dict(),
                            'refined_structure':
                                item['refined_structure'].as_dict(),
                            'scmatrix': item['scmatrix'].tolist(),
                            'features': item['features'].tolist(),
                            'size': item['size']})

        d = {'@module': self.__class__.__module__,
             '@class': self.__class__.__name__,
             '_subspace': self._subspace.as_dict(),
             'items': s_items,
             'metadata': self.metadata}
        return d
