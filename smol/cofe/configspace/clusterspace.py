from __future__ import division
import warnings
import numpy as np
from pymatgen import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher, OrderDisorderElementComparator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SymmOp
from pymatgen.util.coord import is_coord_subset, is_coord_subset_pbc

from . import Orbit, ClusterSupercell, basis_factory
from ..utils import SymmetryError, StructureMatchError, SYMMETRY_ERROR_MESSAGE, SITE_TOL

def get_bits(structure):
    """
    Helper method to compute list of species on each site.
    Includes vacancies
    """
    all_bits = []
    for site in structure:
        bits = []
        for sp in sorted(site.species.keys()):
            bits.append(str(sp))
        if site.species.num_atoms < 0.99:
            bits.append("Vacancy")
        all_bits.append(bits)
    return all_bits


class ClusterSubspace(object):
    """
    Holds a structure, its expansion structure and a list of Orbits. This class defines the Cluster subspace over
    which to fit a cluster expansion: This sets the orbits (groups of clusters) that are to be considered in the fit.

    This is probably the class you're looking for to start defining a cluster expansion.

    You probably want to generate from ClusterSubspace.from_radii, which will auto-generate the orbits,
    unless you want more control over them.
    """

    def __init__(self, structure, expansion_structure, symops, orbits, ltol=0.2, stol=0.1, angle_tol=5,
                 supercell_size='volume'):
        """
        Args:
            structure:
                disordered structure to build a cluster expansion for. Typically the primitive cell
            expansion_structure:
            symops:
            orbits:
            ltol, stol, angle_tol, supercell_size: parameters to pass through to the StructureMatcher.
                Structures that don't match to the primitive cell under these tolerances won't be included
                in the expansion. Easiest option for supercell_size is usually to use a species that has a
                constant amount per formula unit.
        """

        self.stol = stol
        self.ltol = ltol
        self.angle_tol = angle_tol
        self.structure = structure
        self.expansion_structure = expansion_structure
        self.symops = symops

        # test that all the found symmetry operations map back to the input structure
        # otherwise you can get weird subset/superset bugs
        fc = self.structure.frac_coords
        for op in self.symops:
            if not is_coord_subset_pbc(op.operate_multi(fc), fc, SITE_TOL):
                raise SymmetryError(SYMMETRY_ERROR_MESSAGE)

        self.supercell_size = supercell_size

        self.sm = StructureMatcher(primitive_cell=False,
                                   attempt_supercell=True,
                                   allow_subset=True,
                                   scale=True,
                                   supercell_size=self.supercell_size,
                                   comparator=OrderDisorderElementComparator(),
                                   stol=self.stol,
                                   ltol=self.ltol,
                                   angle_tol=self.angle_tol)
        self._orbits = orbits

        # assign the cluster ids
        n_clusters = 1
        n_bit_orderings = 1
        n_orbits = 1
        for k in sorted(self._orbits.keys()):
            for y in self._orbits[k]:
                n_orbits, n_bit_orderings, n_clusters = y.assign_ids(n_orbits, n_bit_orderings, n_clusters)
        self.n_orbits = n_orbits
        self.n_clusters = n_clusters
        self.n_bit_orderings = n_bit_orderings
        self._supercells = {}
        self._external_terms = []

    @classmethod
    def from_radii(cls, structure, radii, ltol=0.2, stol=0.1, angle_tol=5, supercell_size='volume',
                   basis='indicator', orthonormal=False):
        """
        Args:
            structure:
                disordered structure to build a cluster expansion for. Typically the primitive cell
            radii:
                dict of {cluster_size: max_radius}. Radii should be strictly decreasing.
                Typically something like {2:5, 3:4}
            ltol, stol, angle_tol, supercell_size: parameters to pass through to the StructureMatcher.
                Structures that don't match to the primitive cell under these tolerances won't be included
                in the expansion. Easiest option for supercell_size is usually to use a species that has a
                constant amount per formula unit.
        """
        symops = SpacegroupAnalyzer(structure).get_symmetry_operations()
        #get the sites to expand over
        sites_to_expand = [site for site in structure if site.species.num_atoms < 0.99 \
                            or len(site.species) > 1]
        expansion_structure = Structure.from_sites(sites_to_expand)
        orbits = cls._orbits_from_radii(expansion_structure, radii, symops, basis, orthonormal)
        return cls(structure=structure, expansion_structure=expansion_structure, symops=symops, orbits=orbits,
                   ltol=ltol, stol=stol, angle_tol=angle_tol, supercell_size=supercell_size)

    @staticmethod
    def _orbits_from_radii(expansion_structure, radii, symops, basis, orthonormal):
        """
        Generates dictionary of {size: [Orbits]} given a dictionary of maximal cluster radii and symmetry
        operations to apply (not necessarily all the symmetries of the expansion_structure)
        """
        bits = get_bits(expansion_structure)
        nbits = np.array([len(b) - 1 for b in bits])
        sbases = tuple(basis_factory(basis, bit) for bit in bits)
        if orthonormal:
            for basis in sbases:
                basis.orthonormalize()
        orbits = {}
        new_orbits = []

        for bit, nbit, site, sbasis in zip(bits, nbits, expansion_structure, sbases):
            new_orbit = Orbit([site.frac_coords], expansion_structure.lattice,
                              [np.arange(nbit)], [sbasis], symops)
            if new_orbit not in new_orbits:
                new_orbits.append(new_orbit)

        orbits[1] = sorted(new_orbits, key = lambda x: (np.round(x.radius,6), -x.multiplicity))
        all_neighbors = expansion_structure.lattice.get_points_in_sphere(expansion_structure.frac_coords,
                                                                         [0.5, 0.5, 0.5],
                                                                         max(radii.values()) +
                                                                         sum(expansion_structure.lattice.abc)/2)
        for size, radius in sorted(radii.items()):
            new_orbits = []
            for orbit in orbits[size-1]:
                if orbit.radius > radius:
                    continue
                for n in all_neighbors:
                    p = n[0]
                    if is_coord_subset([p], orbit.basecluster.sites, atol=SITE_TOL):
                        continue
                    new_orbit = Orbit(np.concatenate([orbit.basecluster.sites, [p]]), expansion_structure.lattice,
                                      orbit.bits + [np.arange(nbits[n[2]])], orbit.sbases + [sbases[n[2]]], symops)
                    if new_orbit.radius > radius + 1e-8:
                        continue
                    elif new_orbit not in new_orbits:
                        new_orbits.append(new_orbit)

            orbits[size] = sorted(new_orbits, key = lambda x: (np.round(x.radius,6), -x.multiplicity))
        return orbits

    @property
    def external_terms(self):
        return self._external_terms

    def add_external_term(self, term, *args, **kwargs):
        self._external_terms.append((term, args, kwargs))

    def supercell_matrix_from_structure(self, structure):
        sc_matrix = self.sm.get_supercell_matrix(structure, self.structure)
        if sc_matrix is None:
            raise StructureMatchError("Supercell couldn't be found")
        if np.linalg.det(sc_matrix) < 0:
            sc_matrix *= -1
        return sc_matrix

    def supercell_from_structure(self, structure):
        sc_matrix = self.supercell_matrix_from_structure(structure)
        return self.supercell_from_matrix(sc_matrix)

    def supercell_from_matrix(self, sc_matrix):
        scm = tuple(sorted(tuple(s) for s in sc_matrix))
        if scm in self._supercells:
            sc = self._supercells[scm]
        else:
            supercell = self.structure.copy()
            supercell.make_supercell(sc_matrix)
            sc = ClusterSupercell(self, supercell, sc_matrix, get_bits(supercell))
            self._supercells[scm] = sc
        return sc

    def corr_from_structure(self, structure, return_size=False):
        """
        Given a structure, determines which supercell to use, and gets the correlation vector
        """
        sc = self.supercell_from_structure(structure)
        occu = sc.occu_from_structure(structure)
        corr = sc.corr_from_occupancy(occu)

        # get extra terms. This is for the Ewald term
        extras = [term.corr_from_occu(occu, sc, *args, **kwargs)
                  for term, args, kwargs in self._external_terms]
        corr = np.concatenate([corr, *extras])

        if return_size:
            return corr, sc.size
        else:
            return corr

    def refine_structure(self, structure):
        sc = self.supercell_from_structure(structure)
        occu = sc.occu_from_structure(structure)
        return sc.structure_from_occu(occu)

    def corr_from_external(self, structure, sc_matrix, mapping=None):
        sc = self.supercell_from_matrix(sc_matrix) # get clustersupercell
        if mapping != None:
            sc.mapping = mapping
        occu = sc.occu_from_structure(structure)
        return sc.corr_from_occupancy(occu)

    def refine_structure_external(self, structure, sc_matrix):
        sc = self.supercell_from_matrix(sc_matrix)
        occu, mapping = sc.occu_from_structure(structure, return_mapping = True)
        return sc.structure_from_occu(occu), mapping

    @property
    def orbits(self):
        """Returns a list of all orbits sorted by size"""
        return [orbit for k, orbits in sorted(self._orbits.items()) for orbit in orbits]

    def iterorbits(self):
        """Orbit generator, yields orbits"""
        for key in self._orbits.keys():
            for orbit in self._orbits[key]:
                yield orbit

    @classmethod
    def from_dict(cls, d):
        symops = [SymmOp.from_dict(so) for so in d['symops']]
        clusters = {}
        for k, v in d['clusters_and_bits'].items():
            clusters[int(k)] = [Orbit(c[0], c[1], symops) for c in v]
        return cls(structure=Structure.from_dict(d['structure']),
                   expansion_structure=Structure.from_dict(d['expansion_structure']),
                   clusters=clusters, symops=symops,
                   ltol=d['ltol'], stol=d['stol'], angle_tol=d['angle_tol'],
                   supercell_size=d['supercell_size'],
                   #use_ewald=d['use_ewald'], use_inv_r=d['use_inv_r'],
                   )

    def as_dict(self):
        c = {}
        for k, v in self._orbits.items():
            c[int(k)] = [(sc.as_dict(), [list(b) for b in sc.bits]) for sc in v]
        return {'structure': self.structure.as_dict(),
                'expansion_structure': self.expansion_structure.as_dict(),
                'symops': [so.as_dict() for so in self.symops],
                'clusters_and_bits': c,
                'ltol': self.ltol,
                'stol': self.stol,
                'angle_tol': self.angle_tol,
                'supercell_size': self.supercell_size,
                #'use_ewald': self.use_ewald,
                #'use_inv_r': self.use_inv_r,
                '@module': self.__class__.__module__,
                '@class': self.__class__.__name__}

    def __str__(self):
        s = "ClusterBasis: {}\n".format(self.structure.composition)
        for k, v in self._orbits.items():
            s += "    size: {}\n".format(k)
            for z in v:
                s += "    {}\n".format(z)
        return s
