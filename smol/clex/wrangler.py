from __future__ import division
from collections import defaultdict
import logging
import numpy as np
from pymatgen import Structure
from smol.clex.cspace.clusterspace import ClusterSubspace

#TODO StructureWrangler takes training data, checks it (ie does it map, etc) and creates feature matrices and fitting data
# maybe add some convenient checker functions like in daniils script
#TODO make StructureWrangler an MSONable??

class StructureWrangler(object):
    """
    Class that handles (wrangles) input data structures and properties to fit in a cluster expansion.
    This class holds a ClusterSubspace used to compute correlation vectors and produce feauter/design matrices used to
    fit the final ClusterExpansion.
    """

    def __init__(self, clustersubspace, structures, properties, max_ewald=None):
        """
        Fit ECI's to obtain a cluster expansion. This init function takes in all possible arguments,
        but its much simpler to use one of the factory classmethods below,
        e.g. EciGenerator.unweighted

        Args:
            clustersubspace: A ClusterSubspace object
            structures: list of Structure objects
            properties: list of total (non-normalized) properties to
            weights: list of weights for the optimization.
            max_dielectric: constrain the dielectric constant to be positive and below the
                supplied value (note that this is also affected by whether the primitive
                cell is the correct size)
            max_ewald: filter the input structures to only use those with low electrostatic
                energies (no large charge separation in cell). This energy is referenced to the lowest
                value at that composition. Note that this is before the division by the relative dielectric
                 constant and is per primitive cell in the cluster exapnsion -- 1.5 eV/atom seems to be a
                 reasonable value for dielectric constants around 10.
        """
        self.cs = clustersubspace
        self.max_ewald = max_ewald

        # Match all input structures to cluster expansion
        self.items = []
        supercell_matrices, fm_rows = [None] * len(structures), [None] * len(structures)
        #TODO structures and properties should be input as a dict or something, not separate lists.
        for s, e, fm_row in zip(structures, properties, fm_rows):
            try:
                m = self.cs.supercell_matrix_from_structure(s)
                sc = self.cs.supercell_from_matrix(m)
                if fm_row is None:
                    fm_row = sc.corr_from_structure(s)
            except Exception: #TODO too broad never catch ALL exceptions like this
                logging.debug('Unable to match {} with energy {} to supercell'
                              ''.format(s.composition, e))
                if self.cs.supercell_size not in ['volume', 'num_sites', 'num_atoms'] \
                        and s.composition[self.cs.supercell_size] == 0:
                    logging.warning('Specie {} not in {}'.format(self.cs.supercell_size, s.composition))
                continue
            self.items.append({'structure': s,
                               'property': e,
                               'supercell': sc,
                               'features': fm_row,
                               'size': sc.size})

        if self.cs.use_ewald and self.max_ewald is not None:
            if self.cs.use_inv_r:
                raise NotImplementedError('cant use inv_r with max ewald yet')

            min_e = defaultdict(lambda: np.inf)
            for i in self.items:
                c = i['structure'].composition.reduced_composition
                if i['features'][-1] < min_e[c]:
                    min_e[c] = i['features'][-1]

            items = []
            for i in self.items:
                r_e = i['features'][-1] - min_e[i['structure'].composition.reduced_composition]
                if r_e > self.max_ewald:
                    logging.debug('Skipping {} with energy {}, ewald energy is {}'
                                  ''.format(i['structure'].composition, i['energy'], r_e))
                else:
                    items.append(i)
            self.items = items

        logging.info("Matched {} of {} structures".format(len(self.items),
                                                          len(structures)))

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

    @classmethod
    def from_dict(cls, d):
        return cls(clustersubspace=ClusterSubspace.from_dict(d['cluster_subspace']),
                   structures=[Structure.from_dict(s) for s in d['structures']],
                   max_ewald=d.get('max_ewald'))

    def as_dict(self):
        return {'cluster_expansion': self.cs.as_dict(),
                'structures': [s.as_dict() for s in self.structures],
                'max_ewald': self.max_ewald,
                #'feature_matrix': self.feature_matrix.tolist(), #TODO should we keep this to be able to use the from_dict?
                '@module': self.__class__.__module__,
                '@class': self.__class__.__name__}