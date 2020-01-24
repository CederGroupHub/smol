from __future__ import division
from collections import defaultdict
import logging
import warnings
from collections.abc import Sequence
from functools import partial
import numpy as np
from monty.json import MSONable
from pymatgen import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry

from smol.cofe.configspace import EwaldTerm
from smol.cofe.configspace.clusterspace import ClusterSubspace
from smol.cofe.utils import StructureMatchError


def weights_e_above_comp(structures, energies, temperature=2000):
    e_above_comp = _energies_above_composition(structures, energies)
    return np.exp(-e_above_comp / (0.00008617 * temperature))


def weights_e_above_hull(structures, energies, ce_structure, temperature=2000):
    e_above_hull = _energies_above_hull(structures, energies, ce_structure)
    return np.exp(-e_above_hull / (0.00008617 * temperature))


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
class StructureWrangler(MSONable):
    """
    Class that handles (wrangles) input data structures and properties to fit
    in a cluster expansion. This class holds a ClusterSubspace used to compute
    correlation vectors and produce feauter/design matrices used to fit the
    final ClusterExpansion.
    """

    def __init__(self, clustersubspace, weight_type=None):
        """
        This class is meant to take all input training data in the form of
        (structure, property) where the property is usually (lets be honest
        always) the energy for the given structure.
        The class takes care of returning the fitting data as a cluster
        correlation design matrix.
        This class also has methods to check/prepare/filter the data, hence
        the name Wrangler from datawrangler.

        Args:
            clustersubspace (ClusterSubspace):
                A ClusterSubspace object that will be used to fit a
                ClusterExpansion with the provided data.
            weight_type (str or dict):
                str specifying type of weights (i.e. 'hull') OR
                dict with single key specifying the type of weights as above,
                and values being dict of kwargs
        """
        self.cs = clustersubspace
        self.get_weights = {'composition': weights_e_above_comp,
                            'hull': partial(weights_e_above_hull,
                                            ce_structure=self.cs.structure)}
        self.items = []
        self.weight_type = weight_type
        self.weight_kwargs = {}

        if isinstance(weight_type, str):
            self.weight_type = weight_type
        elif isinstance(weight_type, Sequence):
            self.weight_type, self.weight_kwargs = weight_type

    @property
    def structures(self):
        return [i['structure'] for i in self.items]

    @property
    def properties(self):
        return np.array([i['property'] for i in self.items])

    @property
    def normalized_properties(self):
        return self.properties / self.sizes

    @property
    def supercells(self):
        return np.array([i['supercell'] for i in self.items])

    @property
    def feature_matrix(self):
        return np.array([i['features'] for i in self.items])

    @property
    def sizes(self):
        return np.array([i['size'] for i in self.items])

    @property
    def weights(self):
        return np.array([i.get('weight') for i in self.items])

    def add_data(self, data, weights=None, verbose=False):
        """
        Add data to Structure Wrangler, computes correlation vector and if
        successful adds data otherwise it ignores that structure.

        Args
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

        """
        items = []

        for i, (s, p) in enumerate(data):
            try:
                m = self.cs.supercell_matrix_from_structure(s)
                sc = self.cs.supercell_from_matrix(m)
                fm_row = self.cs.corr_from_structure(s)
            except StructureMatchError as e:
                if verbose:
                    print(f'Unable to match {s.composition} with energy {p} to'
                          f' supercell. Throwing out.\n'
                          f'Error Message: {str(e)}.')
                continue
            items.append({'structure': s,
                          'property': p,
                          'supercell': sc,
                          'features': fm_row,
                          'size': sc.size})

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

        self.items += items

    def update_features(self):
        """
        Update the features/feature matrix for the data held. This is useful
        when something is changed in the clustersubspace after
        creating the Wrangler, for example added an Ewald term after creating
        the Wrangler.
        """
        for item in self.items:
            item['features'] = self.cs.corr_from_structure(item['structure'])

    def remove_all_data(self):
        """Removes all data from Wrangler"""
        self.items = []

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
        for term, args, kwargs in self.cs.external_terms:
            if term.__name__ == 'EwaldTerm':
                if 'use_inv_r' in kwargs.keys() and kwargs['use_inv_r']:
                    raise NotImplementedError('cant use inv_r with max_ewald')
                ewald_corr = [i['features'][-1] for i in self.items]
        if ewald_corr is None:
            ewald_corr = []
            for s in self.structures:
                supercell = self.cs.supercell_from_structure(s)
                occu = supercell.occu_from_structure(s)
                ewald_corr.append(EwaldTerm.corr_from_occu(occu, supercell))

        min_e = defaultdict(lambda: np.inf)
        for ecorr, item in zip(ewald_corr, self.items):
            c = item['structure'].composition.reduced_composition
            if ecorr < min_e[c]:
                min_e[c] = ecorr

        items = []
        for ecorr, item in zip(ewald_corr, self.items):
            mine = min_e[item['structure'].composition.reduced_composition]
            r_e = ecorr - mine
            if r_e > max_ewald:
                if verbose:
                    print(f'Skipping {item["structure"].composition} '
                          f'ewald energy is {r_e}')
            else:
                items.append(item)
        self.items = items

    @classmethod
    def from_dict(cls, d):
        """
        Creates Orbit from serialized MSONable dict
        """
        sw = cls(clustersubspace=ClusterSubspace.from_dict(d['cs']))
        sw.items = d['items']
        sw.weight_type = d['weight_type']
        return sw

    def as_dict(self):
        """
        Json-serialization dict representation

        Returns:
            MSONable dict
        """
        d = {'@module': self.__class__.__module__,
             '@class': self.__class__.__name__,
             'cs': self.cs.as_dict(),
             'items': self.items,
             'weight_type': self.weight_type}
        return d
