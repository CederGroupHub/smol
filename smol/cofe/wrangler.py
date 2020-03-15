"""
Implementation of a StructureWrangler and additional functions used to
preprocess and check (wrangle) fitting data of structures and properties.
"""

from __future__ import division
from collections import defaultdict
import warnings
from collections.abc import Sequence
from functools import partial
from multiprocessing import Pool, cpu_count
import numpy as np
from monty.json import MSONable
from pymatgen import Composition, Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from smol.cofe.configspace import EwaldTerm
from smol.cofe.configspace.clusterspace import ClusterSubspace
from smol.exceptions import StructureMatchError
from smol.globals import kB


def weights_e_above_comp(structures, energies, temperature=2000):
    """
    Computes weights for structure energy above the minimum reduced composition
    energy.

    Args:
        structures list(pymatgen.Structure):
            structures corresponding to the given energies.
        energies array:
            energies for given structures.
        temperature (float):
            temperature to used in boltzmann weight

    Returns: weights for each structure.
        array
    """
    e_above_comp = _energies_above_composition(structures, energies)
    return np.exp(-e_above_comp / (kB * temperature))


def weights_e_above_hull(structures, energies, ce_structure, temperature=2000):
    """
    Computes weights for structure energy above the hull of all given
    structures

    Args:
        structures list(pymatgen.Structure):
            structures corresponding to the given energies.
        energies array:
            energies for given structures.
        temperature (float):
            temperature to used in boltzmann weight

    Returns: weights for each structure.
        array
    """
    e_above_hull = _energies_above_hull(structures, energies, ce_structure)
    return np.exp(-e_above_hull / (kB * temperature))


def _energies_above_composition(structures, energies):
    """Computes structure energies above reduced composition"""
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


# TODO should have a dictionary with the applied filters and their parameters
#  to keep track of what has been done
# TODO should allow a bunch of properties saved by keys...

class StructureWrangler(MSONable):
    """
    Class that handles (wrangles) input data structures and properties to fit
    in a cluster expansion. This class holds a ClusterSubspace used to compute
    correlation vectors and produce feature/design matrices used to fit the
    final ClusterExpansion.
    """

    def __init__(self, cluster_subspace, weight_type=None):
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
            weight_type (str or dict):
                str specifying type of weights (i.e. 'hull') OR
                dict with single key specifying the type of weights as above,
                and values being dict of kwargs
        """
        self._subspace = cluster_subspace

        weights_hull_partial = partial(weights_e_above_hull,
                                       ce_structure=self._subspace.structure)
        self.get_weights = {'composition': weights_e_above_comp,
                            'hull': weights_hull_partial}
        self._items = []
        self.weight_type = weight_type
        self.weight_kwargs = {}

        if isinstance(weight_type, str):
            self.weight_type = weight_type
        elif isinstance(weight_type, Sequence):
            self.weight_type, self.weight_kwargs = weight_type

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
        weights = [i.get('weight') for i in self._items]
        if len(weights) > 0 and weights[0] is not None:
            return np.array(weights)

    def add_data(self, data, weights=None, verbose=False, nprocs=1):
        """
        Add data to Structure Wrangler, computes correlation vector and if
        successful adds data otherwise it ignores that structure.

        Args
            data (list):
                list of (structure, property) data
            weights (str, list/tuple or array):
                str specifying type of weights (i.e. 'hull') OR
                list/tuple with two elements (name, kwargs) were name specifies
                the type of weights as above, and kwargs are a dict of
                keyword arguments to obtain the weights OR
                array directly specifying the weights
            verbose (bool):
                if True then print structures that fail in StructureMatcher
            nprocs (int):
                number of processes to run data processing on. 0 defaults to
                maximum number of processors.
        """
        if nprocs == 0 or nprocs > cpu_count():
            nprocs = cpu_count()

        fun = partial(self._process_data, verbose=verbose)
        if nprocs == 1:
            # if only one process do not bother creating a Pool
            items = [item for item in map(fun, data) if item is not None]
        else:
            with Pool(processes=nprocs) as p:
                items = [item for item in p.imap(fun, data, len(data)//nprocs)
                         if item is not None]

        if self.weight_type is not None:
            self._set_weights(items, self.weight_type, **self.weight_kwargs)

        if weights is not None:
            kwargs = {}
            if self.weight_type in self.get_weights.keys():
                warnings.warn(f'Argument <weights> has been provided but the '
                              f'type of weights is already set to '
                              f'\'{self.weight_type}\'. Make sure this is what'
                              f'you want.')
            if isinstance(weights, Sequence) and not isinstance(weights, str):
                weights, kwargs = weights
            self._set_weights(items, weights, **kwargs)

        self._items += items

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

    def _process_data(self, item, verbose):
        """
        Check if the structure for this data item can be matched to the cluster
        subspace prim structure to obtain its supercell matrix, correlation,
        and refined structure.
        """
        structure, property = item
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

    def _set_weights(self, items, weights, **kwargs):
        """Set the weight_type for each data point"""
        # This function is not great since it is implicitly using python
        # pass by reference design....
        if isinstance(weights, str):
            if weights not in self.get_weights.keys():
                msg = f'{weights} is not a valid keyword.' \
                      f'Weights must be one of {self.get_weights.keys()}'
                raise AttributeError(msg)
            self.weight_type = weights
            weights = self.get_weights[weights]([i['structure'] for i in items],  # noqa
                                                [i['property'] for i in items],
                                                **kwargs)
        else:
            self.weight_type = 'external'

        for item, weight in zip(items, weights):
            item['weight'] = weight

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
        sw.weight_type = d['weight_type']
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
             'weight_type': self.weight_type}
        return d
