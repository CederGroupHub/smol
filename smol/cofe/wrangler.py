from __future__ import division
from collections import defaultdict
import logging
import warnings
from functools import partial
import numpy as np
from monty.json import MSONable
from pymatgen import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry

from .configspace.clusterspace import ClusterSubspace
from .utils import StructureMatchError


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
    """Computes energies above hull constructed from phase diagram of given structures"""
    pd = _pd(structures, energies, ce_structure)
    e_above_hull = []
    for s, e in zip(structures, energies):
        e_above_hull.append(pd.get_e_above_hull(PDEntry(s.composition.element_composition, e)))
    return np.array(e_above_hull)


def _pd(structures, energies, cs_structure):
    """
    Generate a phase diagram with the structures and energies
    """
    entries = []

    for s, e in zip(structures, energies):
        entries.append(PDEntry(s.composition.element_composition, e))

    max_e = max(entries, key=lambda e: e.energy_per_atom).energy_per_atom + 1000
    for el in cs_structure.composition.keys():
        entries.append(PDEntry(Composition({el: 1}).element_composition, max_e))

    return PhaseDiagram(entries)


# TODO should have a dictionary with the applied filters and their parameters to keep track of what has been done
class StructureWrangler(MSONable):
    """
    Class that handles (wrangles) input data structures and properties to fit in a cluster expansion.
    This class holds a ClusterSubspace used to compute correlation vectors and produce feauter/design matrices used to
    fit the final ClusterExpansion.
    """


    def __init__(self, clustersubspace, data=None, weights=None):
        """
        This class is meant to take all input training data in the form of (structure, property) where the
        property is usually (lets be honest always) the energy for the given structure.
        The class takes care of returning the fitting data as a cluster correlation design matrix.
        This class also has methods to check/prepare/filter the data, hence the name Wrangler from datawrangler.

        Args:
            clustersubspace (ClusterSubspace):
                A ClusterSubspace object that will be used to fit a ClusterExpansion with the provided data.
            data (list):
                list of (structure, property) data
        """
        self.cs = clustersubspace
        self.get_weights = {'composition': weights_e_above_comp,
                            'hull': partial(weights_e_above_hull, ce_structure=self.cs.structure)}
        self.items, self.weights = [], None

        if data is not None:
            self.add_data(data)

            if isinstance(weights, str) :
                if weights not in self.get_weights.keys():
                    raise AttributeError(f'{weights} is not a valid keyword.'
                                         f'Weights must be one of {self.get_weights.keys()}')
                self.weights = self.get_weights[weights](self.structures, self.properties)
            elif weights is not None:
                self.weights = weights

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

    def add_data(self, data):
        """
        Add data to Structure Wrangler, computes correlation vector and if successful adds data otherwise
        it ignores that structure.

        Args
            data (list):
                list of (structure, property) data
        """
        items = []
        for i, (s, p) in enumerate(data):
            try:
                m = self.cs.supercell_matrix_from_structure(s)
                sc = self.cs.supercell_from_matrix(m)
                fm_row = self.cs.corr_from_structure(s)
            except StructureMatchError as e:
                msg = f'Unable to match {s.composition} with energy {p} to supercell. Throwing out. '
                warnings.warn(msg + f'Error Message: {str(e)}.', RuntimeWarning)
                continue
            except:
                raise
            items.append({'structure': s,
                          'property': p,
                          'supercell': sc,
                          'features': fm_row,
                          'size': sc.size})

        self.items += items
        logging.info(f"Matched {len(items)} of {len(data)} structures")

    #TODO change this so data can be filtered by ewald even if the Ewald term is not used in expansion
    # Curently the clustersubspace must have an ewald term as the very last term.
    def filter_by_ewald(self, max_ewald):
        """
        Filter the input structures to only use those with low electrostatic
        energies (no large charge separation in cell). This energy is referenced to the lowest
        value at that composition. Note that this is before the division by the relative dielectric
        constant and is per primitive cell in the cluster expansion -- 1.5 eV/atom seems to be a
        reasonable value for dielectric constants around 10.

        Args:
            max_ewald (float):
                Ewald threshold
        """
        for term, args, kwargs in self.cs.external_terms:
            if term.__name__ == 'EwaldTerm' and 'use_inv_r' in kwargs.keys():
                if kwargs['use_inv_r']:
                    raise NotImplementedError('cant use inv_r with max_ewald yet')

        min_e = defaultdict(lambda: np.inf)
        for i in self.items:
            c = i['structure'].composition.reduced_composition
            if i['features'][-1] < min_e[c]:
                min_e[c] = i['features'][-1]

        items = []
        for i in self.items:
            r_e = i['features'][-1] - min_e[i['structure'].composition.reduced_composition]
            if r_e > max_ewald:
                logging.debug('Skipping {} with energy {}, ewald energy is {}'
                              ''.format(i['structure'].composition, i['property'], r_e))
            else:
                items.append(i)
        self.items = items

    @classmethod
    def from_dict(cls, d):
        """
        Creates Orbit from serialized MSONable dict
        """
        sw = cls(clustersubspace=ClusterSubspace.from_dict(d['cs']))
        sw.items = d['items']
        sw.weights = d['weights']
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
             'weights': self.weights}
        return d
