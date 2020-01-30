"""
This Module implements the ClusterSubspace class necessary to define the terms
to be included in a cluster expansion. A cluster subspace is a finite set of
clusters, more precisely orbits, that define represent vectors which define
a subspace of the total configurational space of a given lattice system.
"""

from __future__ import division
import numpy as np
from monty.json import MSONable
from pymatgen import Structure, PeriodicSite
from pymatgen.analysis.structure_matcher import StructureMatcher,\
     OrderDisorderElementComparator  # , FrameworkComparator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SymmOp
from pymatgen.util.coord import is_coord_subset, is_coord_subset_pbc, \
    lattice_points_in_supercell, coord_list_mapping_pbc
from smol.cofe.configspace import Orbit
from smol.cofe.configspace.basis import basis_factory
from smol.cofe.configspace.utils import SITE_TOL, get_bits
from smol.exceptions import SymmetryError, StructureMatchError,\
    SYMMETRY_ERROR_MESSAGE


class ClusterSubspace(MSONable):
    """
    Holds a structure, its expansion structure and a list of Orbits.
    This class defines the Cluster subspace over which to fit a cluster
    expansion: This sets the orbits (groups of clusters) that are to be
    considered in the fit.

    This is probably the class you're looking for to start defining the
    structure and cluster terms for your cluster expansion.

    You probably want to generate from ClusterSubspace.from_radii, which will
    auto-generate the orbits, unless you want more control over them.
    """

    def __init__(self, structure, expansion_structure, symops, orbits,
                 **matcher_kwargs):
        """
        Args:
            structure (pymatgen.Structure):
            structure (pymatgen.Structure):
                Structure to define the cluster space. Typically the primitive
                cell. Includes all species regardless of partial occupation.
            expansion_structure (pymatgen.Structure):
                Structure including only sites that will be included in the
                Cluster space.
                (partial occupancy)
            symops (list(pymatgen.Symmop)):
                list of Symmops for the given structure.
            orbits (dict(size: list(Orbit))):
                dictionary with size (number of sites) as keys and list of
                Orbits as values.
            matcher_kwargs:
                ltol, stol, angle_tol, supercell_size: parameters to pass
                through to the StructureMatcher. Structures that don't match to
                the primitive cell under these tolerances won't be included
                in the expansion. Easiest option for supercell_size is usually
                to use a species that has a constant amount per formula unit.
        """

        self.structure = structure
        self.exp_structure = expansion_structure
        self.symops = symops

        # TODO remove this maybe pass the structure matcher as people wanted
        self.structure_matcher_kwargs = matcher_kwargs

        # test that all the found symmetry operations map back to the input
        # structure otherwise you can get weird subset/superset bugs
        fc = self.structure.frac_coords
        for op in self.symops:
            if not is_coord_subset_pbc(op.operate_multi(fc), fc, SITE_TOL):
                raise SymmetryError(SYMMETRY_ERROR_MESSAGE)

        comparator = OrderDisorderElementComparator()
        # Doesn't seem to change success rate for matching structures, but may
        # be a good option to try if it is failing.
        # comparator = FrameworkComparator()

        # This structure matcher is used to determine if a given (supercell)
        # structure matches the prim structure by retrieving the matrix
        # relating them. Only the "get_supercell_matrix" method is used
        self._scmatcher = StructureMatcher(primitive_cell=False,
                                           attempt_supercell=True,
                                           allow_subset=True,
                                           comparator=comparator,
                                           scale=True,
                                           **matcher_kwargs)

        # This structure matcher is used to find the mapping between the sites
        # of a given structure and the sites in the appropriate sized supercell
        # of the prim structure. Only the "get_mapping method" is used
        site_comparator = OrderDisorderElementComparator()
        self._site_matcher = StructureMatcher(primitive_cell=False,
                                              attempt_supercell=False,
                                              allow_subset=True,
                                              comparator=site_comparator,
                                              scale=True,
                                              **matcher_kwargs)

        self._orbits = orbits
        self._supercells = {}
        self._supercell_orb_inds = {}
        self._external_terms = []

        # assign the cluster ids
        n_clstr = 1
        n_bit_ords = 1
        n_orbs = 1
        for k in sorted(self._orbits.keys()):
            for y in self._orbits[k]:
                n_orbs, n_bit_ords, n_clstr = y.assign_ids(n_orbs,
                                                           n_bit_ords,
                                                           n_clstr)
        self.n_orbits = n_orbs
        self.n_clusters = n_clstr
        self.n_bit_orderings = n_bit_ords

    @classmethod
    def from_radii(cls, structure, radii, ltol=0.2, stol=0.1, angle_tol=5,
                   supercell_size='volume', basis='indicator',
                   orthonormal=False):
        """
        Creates a ClusterSubspace with orbits of the given size and radius
        smaller than or equal to the given radius.
        This is the best (and the only easy) way to create one.

        Args:
            structure:
                disordered structure to build a cluster expansion for.
                Typically the primitive cell
            radii:
                dict of {cluster_size: max_radius}. Radii should be strictly
                decreasing. Typically something like {2:5, 3:4}
            ltol, stol, angle_tol, supercell_size: parameters to pass through
                to the StructureMatcher. Structures that don't match to the
                primitive cell under these tolerances won't be included in the
                expansion. Easiest option for supercell_size is usually to use
                a species that has a constant amount per formula unit.
            basis (str):
                a string specifying the site basis functions
            orthonormal (bool):
                wether to enforece an orthonormal basis. From the current
                available bases only the indicator basis is not orthogonal out
                of the box
        Returns:
            ClusterSubSpace
        """

        symops = SpacegroupAnalyzer(structure).get_symmetry_operations()
        # get the sites to expand over
        sites_to_expand = [site for site in structure
                           if site.species.num_atoms < 0.99
                           or len(site.species) > 1]
        expansion_structure = Structure.from_sites(sites_to_expand)
        orbits = cls._orbits_from_radii(expansion_structure, radii, symops,
                                        basis, orthonormal)
        return cls(structure=structure,
                   expansion_structure=expansion_structure, symops=symops,
                   orbits=orbits, ltol=ltol, stol=stol, angle_tol=angle_tol,
                   supercell_size=supercell_size)

    @property
    def orbits(self):
        """Returns a list of all orbits sorted by size"""
        return [orbit for k, orbits
                in sorted(self._orbits.items()) for orbit in orbits]

    def iterorbits(self):
        """Orbit generator, yields orbits"""
        for key in self._orbits.keys():
            for orbit in self._orbits[key]:
                yield orbit

    @property
    def external_terms(self):
        return self._external_terms

    def add_external_term(self, term, *args, **kwargs):
        """
        Add an external term (e.g. an Ewald term) to the cluster expansion
        terms.
        """
        self._external_terms.append((term, args, kwargs))

    def num_prims_from_matrix(self, scmatrix):
        """
        Return the number of prim cells in the super cell corresponding to
        the given matrix
        """
        return int(round(np.abs(np.linalg.det(scmatrix))))

    def corr_from_structure(self, structure, extensive=False):
        """
        Returns the correlation vector for a given structure. To do this the
        correct supercell matrix of the prim necessary needs to be found to
        then determine the mappings between sites to create the occupancy
        vector and also determine the orbit mappings to evaluate the
        corresponding cluster functions.
        """
        scmatrix = self.scmatrix_from_structure(structure)
        occu = self.occupancy_from_structure(structure, scmatrix)
        orb_inds = self.supercell_orbit_mappings(scmatrix)

        corr = self.corr_from_occupancy(occu, orb_inds)

        supercell_size = self.num_prims_from_matrix(scmatrix)

        # get extra terms. This is for the Ewald term
        # The interface for extra terms can be much improved...if anyone cares
        supercell = self.structure.copy()
        supercell.make_supercell(scmatrix)
        extras = [term.corr_from_occu(occu, supercell, orb_inds,
                                      *args, **kwargs)/supercell_size
                  for term, args, kwargs in self._external_terms]
        corr = np.concatenate([corr, *extras])

        if extensive:
            corr *= supercell_size

        return corr

    def refine_structure(self, structure):
        """
        Refine a (relaxed) structure to a perfect supercell structure of the
        the prim structure (aka the corresponding unrelaxed structure)
        """
        scmatrix = self.scmatrix_from_structure(structure)
        occu = self.occupancy_from_structure(structure, scmatrix)

        supercell_structure = self.structure.copy()
        supercell_structure.make_supercell(scmatrix)

        sites = []
        for sp, s in zip(occu, supercell_structure):
            if sp != 'Vacancy':
                site = PeriodicSite(sp, s.frac_coords,
                                    supercell_structure.lattice)
                sites.append(site)
        return Structure.from_sites(sites)

    def occupancy_from_structure(self, structure, scmatrix=None):
        """
        Returns a tuple of occupancies of each site in a the structure in the
        appropriate order set implicitly by the scmatrix that is found
        """
        if scmatrix is None:
            scmatrix = self.scmatrix_from_structure(structure)

        supercell = self.structure.copy()
        supercell.make_supercell(scmatrix)

        mapping = self._structure_site_mapping(supercell, structure)

        occu = []  # np.zeros(len(self.supercell_structure), dtype=np.int)

        for i, bit in enumerate(get_bits(supercell)):
            # rather than starting with all vacancies and looping
            # only over mapping, explicitly loop over everything to
            # catch vacancies on improper sites
            if i in mapping:
                sp = str(structure[mapping.index(i)].specie)
            else:
                sp = 'Vacancy'
            if sp not in bit:
                raise StructureMatchError(f'A site in given structure has a'
                                          f' unrecognized specie {sp}. ')
            occu.append(sp)

        return tuple(occu)

    def scmatrix_from_structure(self, structure):
        """
        Obtain the supercell_structure matrix to convert given structure to
        prim structure.
        """
        scmatrix = self._scmatcher.get_supercell_matrix(structure,
                                                        self.structure)
        if scmatrix is None:
            raise StructureMatchError('Supercell could not be found from '
                                      'structure')
        if np.linalg.det(scmatrix) < 0:
            scmatrix *= -1
        return scmatrix

    def supercell_orbit_mappings(self, scmatrix):
        """
        Return the orbit mappings for a specific supercell of the prim
        structure represented by the given matrix
        """

        # np.arrays are not hashable and can't be used as dict keys.
        scm = tuple(sorted(tuple(s) for s in scmatrix))
        indices = self._supercell_orb_inds.get(scm)

        if indices is None:
            indices = self._gen_orbit_indices(scmatrix)
            self._supercell_orb_inds[scm] = indices

        return indices

    def corr_from_occupancy(self, occu, orbit_indices):
        """
        Each entry in the correlation vector corresponds to a particular
        symmetrically distinct bit ordering
        """
        corr = np.zeros(self.n_bit_orderings)
        corr[0] = 1  # zero point cluster
        occu = np.array(occu)
        for orb, inds in orbit_indices:
            c_occu = occu[inds]
            for i, bit_list in enumerate(orb.bit_combos):
                p = [np.fromiter(map(lambda occu: orb.eval(bits, occu),
                                     c_occu[:]), dtype=np.float)
                     for bits in bit_list]
                corr[orb.orb_b_id + i] = np.concatenate(p).mean()

        return corr

    @staticmethod
    def _orbits_from_radii(expansion_struct, radii, symops, basis,
                           orthonormal):
        """
        Generates dictionary of {size: [Orbits]} given a dictionary of maximal
        cluster radii and symmetry operations to apply (not necessarily all the
        symmetries of the expansion_structure)
        """

        bits = get_bits(expansion_struct)
        nbits = np.array([len(b) - 1 for b in bits])
        sbases = tuple(basis_factory(basis, bit) for bit in bits)
        if orthonormal:
            for basis in sbases:
                basis.orthonormalize()

        orbits = {}
        new_orbits = []

        for bit, nbit, site, sbasis in zip(bits, nbits,
                                           expansion_struct, sbases):
            new_orbit = Orbit([site.frac_coords], expansion_struct.lattice,
                              [np.arange(nbit)], [sbasis], symops)
            if new_orbit not in new_orbits:
                new_orbits.append(new_orbit)

        orbits[1] = sorted(new_orbits, key=lambda x: (np.round(x.radius, 6), -x.multiplicity))  # noqa

        all_neighbors = expansion_struct.lattice.get_points_in_sphere(expansion_struct.frac_coords,  # noqa
                                                                      [0.5, 0.5, 0.5],  # noqa
                                                                      max(radii.values()) + sum(expansion_struct.lattice.abc) / 2)  # noqa
        for size, radius in sorted(radii.items()):
            new_orbits = []
            for orbit in orbits[size-1]:
                if orbit.radius > radius:
                    continue
                for n in all_neighbors:
                    p = n[0]
                    if is_coord_subset([p], orbit.basecluster.sites,
                                       atol=SITE_TOL):
                        continue
                    new_sites = np.concatenate([orbit.basecluster.sites, [p]])
                    new_orbit = Orbit(new_sites, expansion_struct.lattice,
                                      orbit.bits + [np.arange(nbits[n[2]])],
                                      orbit.site_bases + [sbases[n[2]]],
                                      symops)
                    if new_orbit.radius > radius + 1e-8:
                        continue
                    elif new_orbit not in new_orbits:
                        new_orbits.append(new_orbit)

            orbits[size] = sorted(new_orbits, key=lambda x: (np.round(x.radius, 6), -x.multiplicity))  # noqa
        return orbits

    def _structure_site_mapping(self, supercell, structure):
        """
        Returns the mapping between sites in the given structure and a prim
        supercell of the corresponding size
        """

        mapping = self._site_matcher.get_mapping(supercell, structure)
        if mapping is None:
            raise StructureMatchError('Mapping could not be found from '
                                      'structure')
        return mapping.tolist()

    def _gen_orbit_indices(self, scmatrix):
        """
        Finds all the indices associated with each orbit for the supercell
        structure corresponding to the given supercell matrix
        """

        supercell = self.structure.copy()
        supercell.make_supercell(scmatrix)
        prim_to_supercell = np.linalg.inv(scmatrix)
        supercell_fcoords = np.array(supercell.frac_coords)

        ts = lattice_points_in_supercell(scmatrix)
        orbit_indices = []
        for orbit in self.iterorbits():
            prim_fcoords = np.array([c.sites for c in orbit.clusters])
            fcoords = np.dot(prim_fcoords, prim_to_supercell)
            # tcoords contains all the coordinates of the symmetrically
            # equivalent clusters the indices are: [equivalent cluster
            # (primitive cell), translational image, index of site in cluster,
            # coordinate index]
            tcoords = fcoords[:, None, :, :] + ts[None, :, None, :]
            tcs = tcoords.shape
            inds = coord_list_mapping_pbc(tcoords.reshape((-1, 3)),
                                          supercell_fcoords,
                                          atol=SITE_TOL).reshape((tcs[0] * tcs[1], tcs[2]))  # noqa
            # orbit_indices holds orbit, and 2d array of index groups that
            # correspond to the orbit
            # the 2d array may have some duplicates. This is due to
            # symetrically equivalent groups being matched to the same sites
            # (eg in simply cubic all 6 nn interactions will all be [0, 0]
            # indices. This multiplicity disappears as supercell_structure size
            # increases, so I haven't implemented a more efficient method
            orbit_indices.append((orbit, inds))

        return orbit_indices

    def __str__(self):
        s = "ClusterBasis: {}\n".format(self.structure.composition)
        for k, v in self._orbits.items():
            s += "    size: {}\n".format(k)
            for z in v:
                s += "    {}\n".format(z)
        return s

    @classmethod
    def from_dict(cls, d):
        """
        Creates ClusterSubspace from serialized MSONable dict
        """

        symops = [SymmOp.from_dict(so_d) for so_d in d['symops']]
        orbits = {s: [Orbit.from_dict(o) for o in v]
                  for s, v in d['orbits'].items()}
        structure = Structure.from_dict(d['structure'])
        exp_structure = Structure.from_dict(d['expansion_structure'])
        cs = cls(structure=structure,
                 expansion_structure=exp_structure,
                 orbits=orbits, symops=symops,
                 **d['structure_matcher_kwargs'])
        # TODO implement dis
        # subspace._external_terms = [ExternalTerm.from_dict(et_d)
        # for et_d in d['external_terms']]
        return cs

    def as_dict(self):
        """
        Json-serialization dict representation

        Returns:
            MSONable dict
        """

        d = {'@module': self.__class__.__module__,
             '@class': self.__class__.__name__,
             'structure': self.structure.as_dict(),
             'expansion_structure': self.exp_structure.as_dict(),
             'symops': [so.as_dict() for so in self.symops],
             'orbits': {s: [o.as_dict() for o in v]
                        for s, v in self._orbits.items()},
             'structure_matcher_kwargs': self.structure_matcher_kwargs
             # 'external_terms': [et.as_dict() for et in self.external_terms]
             }
        return d
