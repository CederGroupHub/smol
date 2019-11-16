from __future__ import division
from warnings import warn
import numpy as np
from pymatgen import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher, OrderDisorderElementComparator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SymmOp
from pymatgen.util.coord import is_coord_subset, is_coord_subset_pbc

from smol.clex.orbit import Orbit
from .supercell import get_bits, ClusterSupercell
from .utils import SYMMETRY_ERROR, SITE_TOL

#TODO This needs to be renamed to a clusterspace and include only the abstractions defining a clustersubspace

class ClusterSubspace(object):
    """
    Holds lists of Orbits and ClusterSupercells. This class defines the Cluster subspace over which to fit
    a cluster expansion: This sets the orbits (groups of clusters) that are to be considered in the fit.

    This is probably the class you're looking for to start defining a cluster expansion.

    You probably want to generate from ClusterSubspace.from_radii, which will auto-generate the orbits,
    unless you want more control over them.
    """

    def __init__(self, structure, expansion_structure, symops, orbits, ltol=0.2, stol=0.1, angle_tol=5,
                 supercell_size='volume', use_ewald=False, use_inv_r=False, eta=None):
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
                use_ewald:
                    whether to calculate the ewald energy of each structure and use it as a feature. Typically
                    a good idea for ionic materials.
                use_inv_r:
                    experimental feature that allows fitting to arbitrary 1/r interactions between specie-site
                    combinations.
                eta:
                    parameter to override the EwaldSummation default eta. Usually only necessary if use_inv_r=True
            """

        if use_inv_r and eta is None:
            warn("Be careful, you might need to change eta to get properly "
                 "converged electrostatic energies. This isn't well tested")
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
                raise SYMMETRY_ERROR

        self.supercell_size = supercell_size
        self.use_ewald = use_ewald
        self.eta = eta
        self.use_inv_r = use_inv_r
        self.sm = StructureMatcher(primitive_cell=False,
                                   attempt_supercell=True,
                                   allow_subset=True,
                                   scale=True,
                                   supercell_size=self.supercell_size,
                                   comparator=OrderDisorderElementComparator(),
                                   stol=self.stol,
                                   ltol=self.ltol,
                                   angle_tol=self.angle_tol)
        self.orbits = orbits

        # assign the cluster ids
        n_clusters = 1
        n_bit_orderings = 1
        n_sclusters = 1
        for k in sorted(self.orbits.keys()):
            for y in self.orbits[k]:
                n_sclusters, n_bit_orderings, n_clusters = y.assign_ids(n_sclusters, n_bit_orderings, n_clusters)
        self.n_sclusters = n_sclusters
        self.n_clusters = n_clusters
        self.n_bit_orderings = n_bit_orderings
        self._supercells = {}

    @classmethod
    def from_radii(cls, structure, radii, ltol=0.2, stol=0.1, angle_tol=5, supercell_size='volume',
                   use_ewald=False, use_inv_r=False, eta=None):
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
            use_ewald:
                whether to calculate the ewald energy of each structure and use it as a feature. Typically
                a good idea for ionic materials.
            use_inv_r:
                experimental feature that allows fitting to arbitrary 1/r interactions between specie-site
                combinations.
            eta:
                parameter to override the EwaldSummation default eta. Usually only necessary if use_inv_r=True
        """
        symops = SpacegroupAnalyzer(structure).get_symmetry_operations()
        #get the sites to expand over
        sites_to_expand = [site for site in structure if site.species.num_atoms < 0.99 \
                            or len(site.species) > 1]
        expansion_structure = Structure.from_sites(sites_to_expand)
        orbits = cls._orbits_from_radii(expansion_structure, radii, symops)
        return cls(structure=structure, expansion_structure=expansion_structure, symops=symops, orbits=orbits,
                   ltol=ltol, stol=stol, angle_tol=angle_tol, supercell_size=supercell_size, use_ewald=use_ewald,
                   use_inv_r=use_inv_r, eta=eta)

    @classmethod
    def _orbits_from_radii(cls, expansion_structure, radii, symops):
        """
        Generates dictionary of {size: [Orbits]} given a dictionary of maximal cluster radii and symmetry
        operations to apply (not necessarily all the symmetries of the expansion_structure)
        """
        bits = get_bits(expansion_structure)
        nbits = np.array([len(b) - 1 for b in bits])
        orbits = {}
        new_orbits = []

        for i, site in enumerate(expansion_structure):
            orbit = Orbit([site.frac_coords], expansion_structure.lattice, [np.arange(nbits[i])], symops)
            if orbit not in new_orbits:
                new_orbits.append(orbit)
        orbits[1] = sorted(new_orbits, key = lambda x: (np.round(x.max_radius,6), -x.multiplicity))

        all_neighbors = expansion_structure.lattice.get_points_in_sphere(expansion_structure.frac_coords, [0.5, 0.5, 0.5],
                                    max(radii.values()) + sum(expansion_structure.lattice.abc)/2)

        for size, radius in sorted(radii.items()):
            new_orbits = []
            for c in orbits[size-1]:
                if c.max_radius > radius:
                    continue
                for n in all_neighbors:
                    p = n[0]
                    if is_coord_subset([p], c.sites, atol=SITE_TOL):
                        continue
                    orbit = Orbit(np.concatenate([c.sites, [p]]), expansion_structure.lattice,
                                   c.bits + [np.arange(nbits[n[2]])], symops)
                    if orbit.max_radius > radius + 1e-8:
                        continue
                    elif orbit not in new_orbits:
                        new_orbits.append(orbit)

            orbits[size] = sorted(new_orbits, key = lambda x: (np.round(x.max_radius,6), -x.multiplicity))
        return orbits

    def supercell_matrix_from_structure(self, structure):
        sc_matrix = self.sm.get_supercell_matrix(structure, self.structure)
        if sc_matrix is None:
            raise ValueError("Supercell couldn't be found")
        if np.linalg.det(sc_matrix) < 0:
            sc_matrix *= -1
        return sc_matrix

    def supercell_from_structure(self, structure):
        sc_matrix = self.supercell_matrix_from_structure(structure)
        return self.supercell_from_matrix(sc_matrix)

    def supercell_from_matrix(self, sc_matrix):
        sc_matrix = tuple(sorted(tuple(s) for s in sc_matrix))
        if sc_matrix in self._supercells:
            cs = self._supercells[sc_matrix]
        else:
            cs = ClusterSupercell(sc_matrix, self)
            self._supercells[sc_matrix] = cs
        return cs

    def corr_from_structure(self, structure):
        """
        Given a structure, determines which supercell to use,
        and gets the correlation vector
        """
        cs = self.supercell_from_structure(structure)
        return cs.corr_from_structure(structure)

    def refine_structure(self, structure):
        sc = self.supercell_from_structure(structure)
        occu = sc.occu_from_structure(structure)
        return sc.structure_from_occu(occu)

    def structure_energy(self, structure, ecis):
        cs = self.supercell_from_structure(structure)
        return cs.structure_energy(structure, ecis)

    def corr_from_external(self, structure, sc_matrix, mapping=None):
        cs = self.supercell_from_matrix(sc_matrix) # get clustersupercell
        self.cs = cs
        if mapping != None:
            cs.mapping = mapping
        return cs.corr_from_structure(structure)

    def refine_structure_external(self, structure, sc_matrix):
        cs = self.supercell_from_matrix(sc_matrix)
        occu, mapping = cs.occu_from_structure(structure, return_mapping = True)
        return cs.structure_from_occu(occu), mapping

    def structure_energy_from_sc_matrix(self, structure, ecis, sc_matrix):
        cs = self.supercell_from_matrix(sc_matrix)
        return cs.structure_energy(structure, ecis)

    @property
    def orbit(self):
        """
        Yields all symmetrized clusters
        """
        for k in sorted(self.orbits.keys()):
            for c in self.orbits[k]:
                yield c

    def __str__(self):
        s = "ClusterBasis: {}\n".format(self.structure.composition)
        for k, v in self.orbits.items():
            s += "    size: {}\n".format(k)
            for z in v:
                s += "    {}\n".format(z)
        return s

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
                   use_ewald=d['use_ewald'], use_inv_r=d['use_inv_r'],
                   eta=d['eta'])

    def as_dict(self):
        c = {}
        for k, v in self.orbits.items():
            c[int(k)] = [(sc.as_dict(), [list(b) for b in sc.bits]) for sc in v]
        return {'structure': self.structure.as_dict(),
                'expansion_structure': self.expansion_structure.as_dict(),
                'symops': [so.as_dict() for so in self.symops],
                'clusters_and_bits': c,
                'ltol': self.ltol,
                'stol': self.stol,
                'angle_tol': self.angle_tol,
                'supercell_size': self.supercell_size,
                'use_ewald': self.use_ewald,
                'use_inv_r': self.use_inv_r,
                'eta': self.eta,
                '@module': self.__class__.__module__,
                '@class': self.__class__.__name__}