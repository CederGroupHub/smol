from __future__ import division
from collections import defaultdict
import logging
import warnings
import numpy as np
from pymatgen import Structure
from .configspace.clusterspace import ClusterSubspace
from .utils import StructureMatchError


#TODO make StructureWrangler an MSONable??
# TODO should have a dictionary with the applied filters and their parameters to keep track of what has been done

class StructureWrangler(object):
    """
    Class that handles (wrangles) input data structures and properties to fit in a cluster expansion.
    This class holds a ClusterSubspace used to compute correlation vectors and produce feauter/design matrices used to
    fit the final ClusterExpansion.
    """

    def __init__(self, clustersubspace, data=None):
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
        self.items = []
        if data is not None:
            self.add_data(data)

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
        return cls(clustersubspace=ClusterSubspace.from_dict(d['cluster_subspace']),
                   structures=[Structure.from_dict(s) for s in d['structures']],
                   max_ewald=d.get('max_ewald'))

    def as_dict(self):
        return {'cluster_expansion': self.cs.as_dict(),
                'structures': [s.as_dict() for s in self.structures],
                #'max_ewald': self.max_ewald,
                #'feature_matrix': self.feature_matrix.tolist(),
                '@module': self.__class__.__module__,
                '@class': self.__class__.__name__}