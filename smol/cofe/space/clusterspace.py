"""Implementation of ClusterSubspace and related PottsSubspace classes.

The ClusterSubspace class is the workhorse for generating the objects and
information necessary for a cluster expansion.

The PottsSubspace class is an (experimental) class that is similar, but
diverges from the CE mathematic formalism.
"""
# pylint: disable=too-many-lines


import warnings
from collections import namedtuple
from copy import deepcopy
from functools import cached_property
from importlib import import_module
from itertools import chain, groupby

import numpy as np
from monty.dev import deprecated
from monty.json import MSONable, jsanitize
from pymatgen.analysis.structure_matcher import (
    OrderDisorderElementComparator,
    StructureMatcher,
)
from pymatgen.core import PeriodicSite, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SymmOp
from pymatgen.util.coord import (
    coord_list_mapping_pbc,
    is_coord_subset,
    is_coord_subset_pbc,
    lattice_points_in_supercell,
)
from scipy.linalg import block_diag

from smol.cofe.space import (
    Orbit,
    Vacancy,
    basis_factory,
    get_allowed_species,
    get_site_spaces,
)
from smol.cofe.space.basis import IndicatorBasis
from smol.cofe.space.constants import SITE_TOL
from smol.utils.cluster import get_orbit_data
from smol.utils.cluster.container import IntArray2DContainer
from smol.utils.cluster.evaluator import ClusterSpaceEvaluator
from smol.utils.cluster.numthreads import SetNumThreads
from smol.utils.exceptions import (
    SYMMETRY_ERROR_MESSAGE,
    StructureMatchError,
    SymmetryError,
)

__author__ = "Luis Barroso-Luque, William Davidson Richards"

# a named tuple to hold ndarray orbit indices and their corresponding cython containers
OrbitIndices = namedtuple("OrbitIndices", ["arrays", "container"])


class ClusterSubspace(MSONable):
    """ClusterSubspace represents a subspace of functions of configuration.

    A :class:`ClusterSubspace` is the main work horse used in constructing a
    cluster expansion. It is necessary to define the terms to be included in a
    cluster expansion. A cluster subspace holds a finite set of orbits that
    contain symmetrically equivalent clusters. The orbits also contain the set
    of orbit basis functions (also known as correlation functions) that
    represent the terms in the cluster expansion. Taken together the set of all
    orbit functions for all orbits included span a subspace of the total
    function space over the configurational space of a given crystal structure
    system.

    The :class:`ClusterSubspace` also has methods to match fitting structures
    and determine site mappings for supercells of different sizes in order to
    compute correlation vectors (i.e. evaluate the orbit functions for a given
    structure).

    You probably want to generate from :code:`ClusterSubspace.from_cutoffs`,
    which will auto-generate the orbits from diameter cutoffs.

    Attributes:
        symops (list of SymmOp):
            Symmetry operations of structure.
        num_corr_functions (int):
            Total number of correlation functions (orbit basis functions)
            included in the subspace.
        num_orbits (int):
            Total number of crystallographic orbits included in the subspace.
            This includes the empty orbit.
        num_clusters (int):
            Total number of clusters in the primitive cell that are included
            in the subspace.
    """

    num_threads = SetNumThreads("_evaluator")

    def __init__(
        self,
        structure,
        expansion_structure,
        symops,
        orbits,
        supercell_matcher=None,
        site_matcher=None,
        num_threads=None,
        **matcher_kwargs,
    ):
        """Initialize a ClusterSubspace.

        You rarely will need to create a ClusterSubspace using the main
        constructor.
        Look at the class method :code:`from_cutoffs` for the "better" way to
        instantiate a ClusterSubspace.

        Args:
            structure (Structure):
                Structure to define the cluster space. Typically the primitive
                cell. Includes all species regardless of partial occupation.
            expansion_structure (Structure):
                Structure including only sites that will be included in the
                Cluster space (i.e. only those with partial occupancy)
            symops (list of Symmop):
                list of Symmops for the given structure.
            orbits (dict): {size: list of Orbits}
                Dictionary with size (number of sites) as keys and list of
                Orbits as values.
            supercell_matcher (StructureMatcher): optional
                StructureMatcher used to find supercell matrices
                relating the prim structure to other structures. If you pass
                this directly you should know how to set the matcher up,
                otherwise matching your relaxed structures can fail, a lot.
            site_matcher (StructureMatcher): optional
                StructureMatcher used to find site mappings
                relating the sites of a given structure to an appropriate
                supercell of the prim structure . If you pass this directly you
                should know how to set the matcher up, otherwise matching your
                relaxed structures can fail, a lot.
            num_threads (int): optional
                Number of threads to use to compute a correlation vector. Note that
                this is not saved when serializing the ClusterSubspace with the
                as_dict method, so if you are loading a ClusterSubspace from a
                file then make sure to set the number of threads as desired.
            matcher_kwargs:
                ltol, stol, angle_tol, supercell_size: parameters to pass
                through to the StructureMatchers. Structures that don't match
                to the primitive cell under these tolerances won't be included
                in the expansion. Easiest option for supercell_size is usually
                to use a species that has a constant amount per formula unit.
                See pymatgen documentation of :class:`StructureMatcher` for
                more details.
        """
        # keep as private attributes
        self._structure = structure
        self._exp_structure = expansion_structure

        self.symops = symops  # should we even keep this as an attribute?
        self.num_corr_functions = None  # set automattically when assigning ids
        self.num_orbits = None  # same as above
        self.num_clusters = None  # same as above

        # Test that all the found symmetry operations map back to the input
        # structure otherwise you can get weird subset/superset bugs.
        fcoords = self._structure.frac_coords
        for sym_op in self.symops:
            if not is_coord_subset_pbc(
                sym_op.operate_multi(fcoords), fcoords, SITE_TOL
            ):
                raise SymmetryError(SYMMETRY_ERROR_MESSAGE)

        # This structure matcher is used to determine if a given (supercell)
        # structure matches the prim structure by retrieving the matrix
        # relating them. Only the "get_supercell_matrix" method is used.
        if supercell_matcher is None:
            sc_comparator = OrderDisorderElementComparator()
            self._sc_matcher = StructureMatcher(
                primitive_cell=False,
                attempt_supercell=True,
                allow_subset=True,
                comparator=sc_comparator,
                scale=True,
                **matcher_kwargs,
            )
        else:
            self._sc_matcher = supercell_matcher

        # This structure matcher is used to find the mapping between the sites
        # of a given supercell structure and the sites in the appropriate sized
        # supercell of the prim structure. Only "get_mapping" method is used.
        if site_matcher is None:
            site_comparator = OrderDisorderElementComparator()
            self._site_matcher = StructureMatcher(
                primitive_cell=False,
                attempt_supercell=False,
                allow_subset=True,
                comparator=site_comparator,
                scale=True,
                **matcher_kwargs,
            )
        else:
            self._site_matcher = site_matcher

        self._orbits = orbits
        self._external_terms = []  # List will hold external terms (i.e. Ewald)

        # assign the cluster ids
        self._assign_orbit_ids()

        # create evaluator
        self._evaluator = ClusterSpaceEvaluator(
            get_orbit_data(self.orbits), self.num_orbits, self.num_corr_functions
        )
        # set the number of threads to use
        self.num_threads = num_threads

        # Dict to cache orbit index mappings, as OrbitIndices named tuples
        # this prevents doing another structure match with the _site_matcher for
        # structures that have already been matched
        self._supercell_orbit_inds = {}

    @classmethod
    def from_cutoffs(
        cls,
        structure,
        cutoffs,
        basis="indicator",
        orthonormal=False,
        use_concentration=False,
        supercell_matcher=None,
        site_matcher=None,
        num_threads=None,
        **matcher_kwargs,
    ):
        """Create a ClusterSubspace from diameter cutoffs.

        Creates a :class:`ClusterSubspace` with orbits of the given size and
        diameter smaller than or equal to the given value. The diameter of an
        orbit is the maximum distance between any two sites of a cluster of
        that orbit.

        This is the best (and the only easy) way to create a
        :class:`ClusterSubspace`.

        Args:
            structure (Structure):
                disordered structure to build a cluster expansion for.
                Typically the primitive cell
            cutoffs (dict):
                dict of {cluster_size: diameter cutoff}. Cutoffs should be
                strictly decreasing. Typically something like {2:5, 3:4}.
                The empty orbit is always included. Singlets are by default
                included, with the exception below.
                To obtain a subspace with only an empty and singlet terms use
                an empty dict {}, or {1: 1}. Adding a cutoff term for point
                terms, i.e. {1: 0} is useful to exclude point terms. Any other
                value for the cutoff will simply be ignored.
            basis (str):
                a string specifying the site basis functions
            orthonormal (bool):
                whether to enforce an orthonormal basis. From the current
                available bases only the indicator basis is not orthogonal out
                of the box
            use_concentration (bool):
                if True, the concentrations in the prim structure sites will be
                used to orthormalize site bases. This gives a cluster
                subspace centered about the prim composition.
            supercell_matcher (StructureMatcher): optional
                StructureMatcher used to find supercell matrices
                relating the prim structure to other structures. If you pass
                this directly you should know how to set the matcher up,
                otherwise matching your relaxed structures will fail, a lot.
            site_matcher (StructureMatcher): optional
                StructureMatcher used to find site mappings
                relating the sites of a given structure to an appropriate
                supercell of the prim structure . If you pass this directly you
                should know how to set the matcher up, otherwise matching your
                relaxed structures will fail, a lot.
            num_threads (int): optional
                Number of threads to use to compute a correlation vector. Note that
                this is not saved when serializing the ClusterSubspace with the
                as_dict method, so if you are loading a ClusterSubspace from a
                file then make sure to set the number of threads as desired.
            matcher_kwargs:
                ltol, stol, angle_tol, supercell_size: parameters to pass
                through to the StructureMatchers. Structures that don't match
                to the primitive cell under these tolerances won't be included
                in the expansion. Easiest option for supercell_size is usually
                to use a species that has a constant amount per formula unit.

        Returns:
            ClusterSubspace
        """
        # get symmetry operations of prim structure.
        symops = SpacegroupAnalyzer(structure).get_symmetry_operations()
        # get the active sites (partial occupancy) to expand over.
        sites_to_expand = [
            site
            for site in structure
            if site.species.num_atoms < 0.99 or len(site.species) > 1
        ]
        expansion_structure = Structure.from_sites(sites_to_expand)
        # get orbits within given cutoffs
        orbits = cls._gen_orbits_from_cutoffs(
            expansion_structure, cutoffs, symops, basis, orthonormal, use_concentration
        )
        return cls(
            structure=structure,
            expansion_structure=expansion_structure,
            symops=symops,
            orbits=orbits,
            supercell_matcher=supercell_matcher,
            site_matcher=site_matcher,
            num_threads=num_threads,
            **matcher_kwargs,
        )

    @property
    def evaluator(self):
        """Get the instance of cluster space evaluator extension type.

        The evaluator is used to compute correlations quickly. You should not use this
        directly, instead use the :meth:`corr_from_structure` method. If you do attempt
        to use directly make sure you understand the code, otherwise you will crash
        your python interpreter. You have been warned...
        """
        return self._evaluator

    @property
    def basis_type(self):
        """Get the type of site basis set used."""
        return self.orbits[0].basis_type

    @property
    def structure(self):
        """Get the underlying primitive structure including inactive sites."""
        return self._structure

    @property
    def expansion_structure(self):
        """Get the primitive expansion structure (excludes inactive sites)."""
        return self._exp_structure

    @property
    def cutoffs(self):
        """Return dict of orbit cluster cutoffs.

        These are "tight" cutoffs, as in the maximum diameter for each cluster
        size, which is <= the input to from_cutoffs.
        """
        return {
            size: max(orbit.base_cluster.diameter for orbit in orbits)
            for size, orbits in self._orbits.items()
            if size != 1
        }

    @property
    def orbits(self):
        """Return a list of all orbits sorted by size."""
        return [orbit for _, orbits in sorted(self._orbits.items()) for orbit in orbits]

    @property
    def orbits_by_size(self):
        """Get dictionary of orbits with key being the orbit size."""
        return self._orbits

    @cached_property
    def orbits_by_diameter(self):
        """Get dictionary of orbits with key being the orbit diameter.

        Diameters are rounded to 6 decimal places.
        """
        return {
            size: tuple(orbits)
            for size, orbits in groupby(
                sorted(
                    self.orbits, key=lambda orb: np.round(orb.base_cluster.diameter, 6)
                ),
                key=lambda orb: np.round(orb.base_cluster.diameter, 6),
            )
        }

    @property
    def orbit_multiplicities(self):
        """Get the crystallographic multiplicities for each orbit."""
        mults = [1] + [orb.multiplicity for orb in self.orbits]
        return np.array(mults)

    @property
    def num_functions_per_orbit(self):
        """Get the number of correlation functions for each orbit.

        The list returned has length equal to the total number of orbits,
        and each entry is the total number of correlation functions
        associated with that orbit.
        """
        return np.array([len(orbit) for orbit in self.orbits])

    @property
    def function_orbit_ids(self):
        """Get Orbit IDs corresponding to each correlation function.

        If the ClusterSubspace includes external terms, these are not included
        in the list since they are not associated with any orbit.
        """
        func_orb_ids = [0]
        for orbit in self.orbits:
            func_orb_ids += len(orbit) * [
                orbit.id,
            ]
        return np.array(func_orb_ids)

    @property
    def function_inds_by_size(self):
        """Get correlation function indices by cluster sizes."""
        return {
            s: list(range(os[0].bit_id, os[-1].bit_id + len(os[-1])))
            for s, os in self._orbits.items()
        }

    @property
    def function_ordering_multiplicities(self):
        """Get array of ordering multiplicity of each correlation function.

        The length of the array returned is the total number of correlation
        functions in the subspace for all orbits. The ordering multiplicity of
        a correlation function is the number of symmetrically equivalent bit
        orderings (function-labeled orbit configurations) that result in the
        product of the same single site functions.
        """
        mults = [1] + [
            mult for orb in self.orbits for mult in orb.bit_combo_multiplicities
        ]
        return np.array(mults)

    @property
    def function_total_multiplicities(self):
        """Get array of total multiplicity of each correlation function.

        The length of the array returned is the total number of correlation
        functions in the subspace for all orbits. The total multiplicity of a
        correlation function is the number of symmetrically equivalent bit
        orderings (or function-labeled orbit configurations) that result in the
        product of the same single site functions times the (crystallographic)
        multiplicity of the orbit.
        """
        return (
            self.orbit_multiplicities[self.function_orbit_ids]
            * self.function_ordering_multiplicities
        )

    @property
    def basis_orthogonal(self):
        """Check if the orbit basis is orthogonal."""
        return all(orb.basis_orthogonal for orb in self.orbits)

    @property
    def basis_orthonormal(self):
        """Check if the orbit basis is orthonormal."""
        return all(orb.basis_orthonormal for orb in self.orbits)

    @property
    def external_terms(self):
        """Get external terms to be fitted together with the correlations.

        External terms are those represented by pair interaction Hamiltonians
        (i.e. Ewald electrostatics).
        """
        return self._external_terms

    @property
    def site_rotation_matrix(self):
        """Get change of basis matrix from site function rotations.

        Note: this is meant only for rotations using orthonormal site bases.
        Using it otherwise will not work as expected.
        """
        return block_diag([1], *[orb.rotation_array for orb in self.orbits])

    def orbits_by_cutoffs(self, upper, lower=0):
        """Get orbits with clusters within given diameter cutoffs (inclusive).

        Args:
           upper (float):
               upper diameter for clusters to include.
           lower (float): optional
               lower diameter for clusters to include.

        Returns:
           list of Orbits
        """
        return [
            orbit
            for orbit in self.iterorbits()
            if lower <= orbit.base_cluster.diameter <= upper
        ]

    def orbit_hierarchy(self, level=1, min_size=1):
        """Get orbit hierarchy by IDs.

        The orbit hierarchy represents in inclusion relationships between orbits and
        their suborbits.

        Args:
            level (int): optional
                how many levels down to look for suborbits. If all suborbits
                are needed make level large enough or set to None.
            min_size (int): optional
                minimum size of clusters in sub orbits to include

        Returns:
            list of list: each element of the inner lists is the orbit id for
            all suborbits corresponding to the orbit at the given outer list
            index.
        """
        sub_ids = [
            [
                suborb.id
                for suborb in self.get_sub_orbits(
                    orb.id, level=level, min_size=min_size
                )
            ]
            for orb in self.orbits
        ]

        return [
            [],
        ] + sub_ids

    def function_hierarchy(self, level=1, min_size=2, invert=False):
        """Get the correlation function hierarchy.

        The function hierarchy is the relationship between specific correlation
        functions and "sub" correlation functions (i.e. a correlation function is
        a "sub" correlation factor or included in higher degree correlation function if
        it is a factor of a higher degree correlation function.

        Args:
            level (int): optional
                how many levels down to look for suborbits. If all suborbits
                are needed make level large enough or set to None.
            min_size (int): optional
                minimum size of clusters in sub orbits to include
            invert (bool): optional
                Default is invert=False which gives the high to low bit combo
                hierarchy. Invert= True will invert the hierarchy into low to
                high

        Returns:
            list of list: each element of the inner lists is the bit id for
            all correlation functions corresponding to the corr function at the given
            outer list index.
        """
        hierarchy = [
            self.get_sub_function_ids(i, level=level, min_size=min_size)
            for i in range(self.num_corr_functions)
        ]

        if invert:
            hierarchy = invert_mapping(hierarchy)

        return hierarchy

    def orbits_from_cutoffs(self, upper, lower=0):
        """Get orbits with clusters within given diameter cutoffs (inclusive).

        Args:
            upper (float or dict):
                upper diameter for clusters to include. If a single float
                is given then that cutoff is used for all orbit sizes.
                Otherwise a dict can be used to specify the cutoff for the
                orbit cluster sizes,
                i.e. {2: pair_cutoff, 3: triplet_cutoff, ...}
            lower (float): optional
                lower diameter for clusters to include. If a single float
                is given then that cutoff is used for all orbit sizes.
                Otherwise a dict can be used to specify the cutoff for the
                orbit cluster sizes,
                i.e. {2: pair_cutoff, 3: triplet_cutoff, ...}

        Returns:
            list of Orbits
        """
        upper = (
            upper
            if isinstance(upper, dict)
            else {k: upper for k in self._orbits.keys()}
        )
        lower = (
            lower
            if isinstance(lower, dict)
            else {k: lower for k in self._orbits.keys()}
        )
        return [
            orbit
            for size in upper.keys()
            for orbit in self._orbits[size]
            if lower[size] <= orbit.base_cluster.diameter <= upper[size]
        ]

    def function_inds_from_cutoffs(self, upper, lower=0):
        """Get indices of correlation functions by cluster cutoffs.

        Args:
            upper (float or dict):
                upper diameter for clusters to include. If a single float
                is given then that cutoff is used for all orbit sizes.
                Otherwise a dict can be used to specify the cutoff for the
                orbit cluster sizes,
                i.e. {2: cutoff_pairs, 3: cutoff_trips, ...}
            lower (float): optional
                lower diameter for clusters to include. If a single float
                is given then that cutoff is used for all orbit sizes.
                Otherwise a dict can be used to specify the cutoff for the
                orbit cluster sizes,
                i.e. {2: cutoff_pairs, 3: cutoff_trips, ...}

        Returns:
            list: list of correlation function indices for clusters within cutoffs
        """
        orbits = self.orbits_from_cutoffs(upper, lower)
        inds = []
        for orbit in orbits:
            inds += list(range(orbit.bit_id, orbit.bit_id + len(orbit)))
        return np.array(inds)

    def add_external_term(self, term):
        """Add an external term to the ClusterSubspace.

        Adds an external term (e.g. an Ewald term) to the cluster expansion
        terms. External term classes must be MSONable and implement a method
        to obtain a "correlation". See smol.cofe.extern for notebooks.

        Args:
            term (ExternalTerm):
                An instance of an external term. Currently only EwaldTerm is
                implemented.
        """
        for added_term in self.external_terms:
            if isinstance(term, type(added_term)):
                raise ValueError(f"This ClusterSubspaces already has an {type(term)}.")
        self._external_terms.append(term)

    @staticmethod
    def num_prims_from_matrix(scmatrix):
        """Get number of prim structures in a supercell for a given matrix."""
        return int(round(np.abs(np.linalg.det(scmatrix))))

    def corr_from_structure(
        self, structure, normalized=True, scmatrix=None, site_mapping=None
    ):
        """Get correlation vector for structure.

        Returns the correlation vector for a given structure. To do this, the
        correct supercell matrix of the prim needs to be found to then
        determine the mappings between sites to create the occupancy
        string and also determine the orbit mappings to evaluate the
        corresponding cluster functions.

        Args:
            structure (Structure):
                Structure to compute correlation from
            normalized (bool):
                return the correlation vector normalized by the prim cell size.
                In theory correlation vectors are always normalized, but
                getting them without normalization allows to compute the
                "extensive" values.
            scmatrix (ndarray): optional
                supercell matrix relating the prim structure to the given
                structure. Passing this if it has already been matched will
                make things much quicker. You are responsible that the
                supercell matrix is correct.
            site_mapping (list): optional
                Site mapping as obtained by
                :code:`StructureMatcher.get_mapping`
                such that the elements of site_mapping represent the indices
                of the matching sites to the prim structure. If you pass this
                option, you are fully responsible that the mappings are correct!

        Returns:
            array: correlation vector for given structure
        """
        if scmatrix is None:
            scmatrix = self.scmatrix_from_structure(structure)

        occu = self.occupancy_from_structure(
            structure, scmatrix=scmatrix, site_mapping=site_mapping, encode=True
        )
        indices = self.get_orbit_indices(scmatrix)
        corr = self._evaluator.correlations_from_occupancy(occu, indices.container)
        size = self.num_prims_from_matrix(scmatrix)

        if self.external_terms:
            supercell = self.structure.copy()
            supercell.make_supercell(scmatrix)
            extras = [
                term.value_from_occupancy(occu, supercell) / size
                for term in self._external_terms
            ]
            corr = np.concatenate([corr, *extras])

        if not normalized:
            corr *= size

        return corr

    def refine_structure(self, structure, scmatrix=None, site_mapping=None):
        """Refine a (relaxed) structure.

        Refine a (relaxed) structure to a perfect supercell structure of the
        the prim structure (aka the corresponding "unrelaxed" structure).

        Args:
            structure (Structure):
                structure to refine to a perfect multiple of the prim
            scmatrix (ndarray): optional
                supercell matrix relating the prim structure to the given
                structure. Passing this if it has already been matched will
                make things much quicker. You are responsible for correctness.
            site_mapping (list): optional
                site mapping as obtained by StructureMatcher.get_mapping
                such that the elements of site_mapping represent the indices
                of the matching sites to the prim structure. Again you are
                responsible.

        Returns:
             Structure: the refined structure
        """
        if scmatrix is None:
            scmatrix = self.scmatrix_from_structure(structure)

        occu = self.occupancy_from_structure(
            structure, scmatrix=scmatrix, site_mapping=site_mapping
        )

        supercell_structure = self.structure.copy()
        supercell_structure.make_supercell(scmatrix)

        sites = []
        for specie, site in zip(occu, supercell_structure):
            if not isinstance(specie, Vacancy):  # skip vacancies
                site = PeriodicSite(
                    specie, site.frac_coords, supercell_structure.lattice
                )
                sites.append(site)
        return Structure.from_sites(sites)

    def occupancy_from_structure(
        self, structure, scmatrix=None, site_mapping=None, encode=False
    ):
        """Occupancy string for a given structure.

        Returns a list of occupancies of each site in the structure in the
        appropriate order set implicitly by the supercell matrix that is found.

        This function is used as input to compute correlation vectors for the
        given structure.

        This function is also useful to obtain an initial occupancy for a Monte
        Carlo simulation. (Make sure that the same supercell matrix is being
        used here as in the instance of the processor class for the simulation.
        Although it is recommended to use the similar function in Processor
        classes.)

        Args:
            structure (Structure):
                structure to obtain a occupancy string for
            scmatrix (array): optional
                supercell matrix relating the given structure and the
                primitive structure. If you pass the supercell, you fully are
                responsible that it is the correct one! This prevents running
                the _scmatcher (supercell structure matcher)
            site_mapping (list): optional
                site mapping as obtained by StructureMatcher.get_mapping
                such that the elements of site_mapping represent the indices
                of the matching sites to the prim structure. If you pass this
                option, you are fully responsible that the mappings are correct!
                This prevents running _site_matcher to get the mappings.
            encode (bool): optional
                if True, the occupancy string will have the index of the species
                in the expansion structure site spaces, rather than the
                species itself.

        Returns:
            list: occupancy string for structure.
        """
        if scmatrix is None:
            scmatrix = self.scmatrix_from_structure(structure)

        supercell = self.structure.copy()
        supercell.make_supercell(scmatrix)

        if site_mapping is None:
            site_mapping = self.structure_site_mapping(supercell, structure)

        occu = []  # np.zeros(len(self.supercell_structure), dtype=int)

        for i, allowed_species in enumerate(get_allowed_species(supercell)):
            # rather than starting with all vacancies and looping
            # only over mapping, explicitly loop over everything to
            # catch vacancies on improper sites
            if i in site_mapping:
                spec = structure[site_mapping.index(i)].specie
            else:
                spec = Vacancy()
            if spec not in allowed_species:
                raise StructureMatchError(
                    "A site in given structure has an  unrecognized species " f"{spec}."
                )
            if encode:
                occu.append(allowed_species.index(spec))
            else:
                occu.append(spec)

        if encode:  # cast to ndarray dtype int
            occu = np.array(occu, dtype=int)

        return occu

    def scmatrix_from_structure(self, structure):
        """Get supercell matrix from a given structure.

        Obtain the supercell structure matrix to convert the prim structure
        to a supercell equivalent to given structure.

        Args:
            structure (Structure):
                a pymatgen Structure.

        Returns:
            ndarray: matrix relating given structure and prim structure.
        """
        scmatrix = self._sc_matcher.get_supercell_matrix(structure, self.structure)
        if scmatrix is None:
            raise StructureMatchError("Supercell could not be found from structure")
        if np.linalg.det(scmatrix) < 0:
            scmatrix *= -1
        return scmatrix

    def supercell_orbit_mappings(self, scmatrix):
        """Get orbit mappings for a structure from supercell of prim.

        Return the orbit mappings for a specific supercell of the prim
        structure represented by the given matrix

        Args:
            scmatrix (array):
                array relating a supercell with the primitive matrix

        Returns:
            tuple of ndarray:
                tuple of 2D ndarrays where each array has the site indices for all
                equivalent orbits in a supercell obtained from the given matrix.
                First dimension are clusters and 2nd dimensiuon are site indices for
                that cluster
        """
        orbit_indices = self.get_orbit_indices(scmatrix)
        return orbit_indices.arrays

    def get_aliased_orbits(self, sc_matrix):
        """Get the aliased orbits for a given supercell shape.

        Detect the orbits that will be aliased due to translational symmetry imposed by
        the supercell lattice. Orbits i and j are aliased when a geometric cluster in
        orbit i is identically mapped to another geometric cluster in orbit j.
        It can be shown through a group theoretical argument that any cluster in orbit i
        then must be identical to a corresponding cluster in orbit j.

        The implication of aliasing is that correlation functions of these orbits will
        evaluate to the same value, leading to feature matrix rank deficiency and
        potentially unphysical ECI.

        This method will detect most cases of orbit degeneracy, but not some edge cases.

        Args:
            sc_matrix: (array):
                array relating a supercell with the primitive matrix

        Returns:
            list of tuples:
                (orbit_id i, orbit_id j, ...) list of tuples containing the orbits
                that are aliased.

        """
        sc_orb_map = self.supercell_orbit_mappings(sc_matrix)
        aliased_orbits = []
        for orb_i, orb_map_i in enumerate(sc_orb_map):
            # +1 because ECI index takes the null cluster as index 0.
            if orb_i + 1 in chain(*aliased_orbits):
                continue
            orb_i_id = orb_i + 1
            aliased = False
            orbit_i_aliased = [orb_i_id]
            sorted_orb_map_i = {tuple(sorted(c_map)) for c_map in orb_map_i}

            for orb_j, orb_map_j in enumerate(sc_orb_map):
                if orb_i >= orb_j or (orb_j + 1 in chain(*aliased_orbits)):
                    continue
                orb_j_id = orb_j + 1
                sorted_orb_map_j = {tuple(sorted(c_map)) for c_map in orb_map_j}

                if sorted_orb_map_i == sorted_orb_map_j:
                    aliased = True
                    orbit_i_aliased.append(orb_j_id)

            orbit_i_aliased = tuple(sorted(orbit_i_aliased))
            if aliased:
                aliased_orbits.append(orbit_i_aliased)

        aliased_orbits = sorted(list(set(aliased_orbits)), key=lambda x: x[0])
        return aliased_orbits

    def change_site_bases(self, new_basis, orthonormal=False):
        """Change the type of site basis used in the site basis functions.

        Args:
            new_basis (str):
                name of new basis for all site bases
            orthonormal (bool):
                option to orthonormalize all new site basis sets
        """
        for orbit in self.orbits:
            orbit.transform_site_bases(new_basis, orthonormal)
        # rest the evaluator
        self._evaluator.reset_data(
            get_orbit_data(self.orbits), self.num_orbits, self.num_corr_functions
        )

    def rotate_site_basis(self, singlet_id, angle, index1=0, index2=1):
        """Apply a rotation to a site basis.

        The rotation is applied around an axis normal to the span of the two
        site functions given by index1 and index 2 (the constant function is
        not included, i.e. index 0 corresponds to the first non constant
        function)

        Read warnings in SiteBasis.rotate when using this method.
        TLDR: Careful when using this with non-orthogonal or biased site bases.

        Args:
            singlet_id (int):
                Orbit id of singlet function. Only singlet function ids are
                valid here.
            angle (float):
                Angle to rotate in radians.
            index1 (int):
                index of first basis vector in function_array
            index2 (int):
                index of second basis vector in function_array
        """
        if singlet_id not in range(1, len(self._orbits[1]) + 1):
            raise ValueError("Orbit id provided is not a valid singlet id.")

        basis = self.orbits[singlet_id - 1].site_bases[0]
        basis.rotate(angle, index1, index2)
        rotated = [basis]
        for orbit in self.orbits:
            for site_basis in orbit.site_bases:
                if (
                    site_basis.site_space == basis.site_space
                    and site_basis not in rotated
                ):  # maybe clean this up?
                    site_basis.rotate(angle, index1, index2)
                    rotated.append(site_basis)
            orbit.reset_bases()
        # rest the evaluator
        self._evaluator.reset_data(
            get_orbit_data(self.orbits), self.num_orbits, self.num_corr_functions
        )

    def remove_orbits(self, orbit_ids):
        """Remove whole orbits by their ids.

        Remove orbits from cluster spaces. It is helpful to print a ClusterSubspace or
        ClusterExpansion to obtain orbit ids. After removing orbits, orbit ID's and
        orbit bit ID's are re-assigned.

        This is useful to prune a ClusterExpansion by removing orbits with small
        associated coefficients or ECI. Note that this will remove a full orbit,
        which for the case of sites with only two species is the same as removing a
        single correlation vector element (only one ECI). For cases with sites having
        more than 2 species allowed per site there is more than one orbit functions
        (for all the possible bit orderings or function- labeled orbit configurations)
        and removing an orbit will remove more than one element in the correlation
        vector.

        Args:
            orbit_ids (list):
                list of orbit ids to be removed
        """
        if min(orbit_ids) < 0:
            raise ValueError("Index out of range. Negative inds are not allowed.")

        if min(orbit_ids) == 0:
            raise ValueError(
                "The empty orbit can not be removed. \n If you really want to "
                "do this remove the first column in your feature matrix before"
                " fitting."
            )
        if max(orbit_ids) > self.num_orbits - 1:
            raise ValueError(
                "Index out of range. " "Total number of orbits is: {self.num_orbits}"
            )

        for size, orbits in self._orbits.items():
            self._orbits[size] = [
                orbit for orbit in orbits if orbit.id not in orbit_ids
            ]

        # remove any empty keys if all orbits of a given size were removed
        for size in list(
            self._orbits.keys()
        ):  # cast to list bc .keys() behaves like an iterator
            if len(self._orbits[size]) == 0:
                del self._orbits[size]

        self._assign_orbit_ids()  # Re-assign ids
        # Clear the cached supercell orbit mappings
        # TODO instead of resetting this, just remove the orbit ids
        self._supercell_orbit_inds = {}
        # reset the evaluator
        self._evaluator.reset_data(
            get_orbit_data(self.orbits), self.num_orbits, self.num_corr_functions
        )
        # clear the cached orbits_by_diameter
        if hasattr(self, "orbits_by_diameter"):
            delattr(self, "orbits_by_diameter")

    def remove_corr_functions(self, corr_ids):
        """Remove correlation functions by their ID's.

        This allows more granular removal of terms involved in fitting/evaluating a
        cluster expansion. Similar to remove_orbits this is useful to prune a cluster
        expansion and actually allows to remove a single term (ie one with small
        associated coefficient/ECI).

        This procedure is perfectly well posed mathematically. The resultant
        CE is still a valid function of configurations with all the necessary
        symmetries from the underlying structure. It is also practically justified
        if we allow "in group" orbit eci sparsity...which everyone in the field
        does anyway. In terms of physical/chemical interpretation it is not
        obvious what it means to remove certain combinations of an n-body
        interaction term, and not the whole term itself...so tread lighlty with your
        model interpretations.

        Args:
            corr_ids (list):
                list of correlation function ids to remove
        """
        empty_orbit_ids = []
        corr_ids = np.array(corr_ids, dtype=int)

        for orbit in self.orbits:
            first_id = orbit.bit_id
            last_id = orbit.bit_id + len(orbit)
            to_remove = corr_ids[corr_ids >= first_id]
            to_remove = to_remove[to_remove < last_id] - first_id
            if to_remove.size > 0:
                try:
                    orbit.remove_bit_combos_by_inds(to_remove)
                except RuntimeError:
                    empty_orbit_ids.append(orbit.id)
                    warnings.warn(
                        "All bit combos have been removed from orbit with id "
                        f"{orbit.id}. This orbit will be fully removed."
                    )

        if empty_orbit_ids:
            # ids are reassigned and evaluator reset in remove_orbits
            self.remove_orbits(empty_orbit_ids)
        else:
            self._assign_orbit_ids()  # Re-assign ids
            self._evaluator.reset_data(
                get_orbit_data(self.orbits), self.num_orbits, self.num_corr_functions
            )

    def copy(self):
        """Deep copy of instance."""
        return ClusterSubspace.from_dict(self.as_dict())

    def structure_site_mapping(self, supercell, structure):
        """Get structure site mapping.

        Returns the mapping between sites in the given structure and a prim
        supercell of the corresponding size.

        Args:
            supercell (Structure):
                supercell of prim structure with same size as other structure.
            structure (Structure):
                Structure to obtain site mappings to supercell of prim
        Returns:
            list: site mappings of structure to supercell
        """
        mapping = self._site_matcher.get_mapping(supercell, structure)
        if mapping is None:
            raise StructureMatchError("Mapping could not be found from structure.")
        return mapping.tolist()

    def get_sub_orbits(self, orbit_id, level=1, min_size=1):
        """Get sub orbits of the orbit for the corresponding orbit_id.

        Args:
            orbit_id (int):
                id of orbit to get sub orbit id for
            level (int): optional
                how many levels down to look for suborbits. If all suborbits
                are needed make level large enough or set to None.
            min_size (int): optional
                minimum size of clusters in sub orbits to include

        Returns:
            list of ints: list containing ids of suborbits
        """
        if orbit_id == 0:
            return []
        size = len(self.orbits[orbit_id - 1].base_cluster)
        if level is None or level < 0 or size - level - 1 < 0:
            stop = 0
        elif min_size > size - level:
            stop = min_size - 1
        else:
            stop = size - level - 1

        search_sizes = range(size - 1, stop, -1)
        return [
            orbit
            for s in search_sizes
            for orbit in self._orbits[s]
            if self.orbits[orbit_id - 1].is_sub_orbit(orbit)
        ]

    def get_sub_function_ids(self, corr_id, level=1, min_size=1):
        """Get the bit combo ids of all sub correlation functions.

        A sub correlation function of a given correlation function means that
        the sub correlation function is a factor of the correlation function
        (with the additional requirement of acting over the sites in sub
        clusters of the clusters over which the given corr function acts on).

        In other words, think of it as an orbit of function-labeled subclusters
        of a given orbit of function-labeled clusters...a mouthful...

        Args:
            corr_id (int):
                id of orbit to get sub orbit id for
            level (int): optional
                how many levels down to look for suborbits. If all suborbits
                are needed make level large enough or set to None.
            min_size (int): optional
                minimum size of clusters in sub orbits to include

        Returns:
            list of ints: list containing ids of sub correlation functions
        """
        if corr_id == 0:
            return []

        orbit = self.orbits[self.function_orbit_ids[corr_id] - 1]
        bit_combo = orbit.bit_combos[corr_id - orbit.bit_id]

        sub_fun_ids = []
        for sub_orbit in self.get_sub_orbits(orbit.id, level=level, min_size=min_size):
            inds = orbit.sub_orbit_mappings(sub_orbit)
            for i, sub_bit_combo in enumerate(sub_orbit.bit_combos):
                if np.any(np.all(sub_bit_combo[0] == bit_combo[:, inds], axis=2)):
                    sub_fun_ids.append(sub_orbit.bit_id + i)

        return sub_fun_ids

    @deprecated(
        message=(
            "This function is deprecated and will be removed in version 0.4.0.\n"
            "You should not have really been using this function anyway..."
        )
    )
    def gen_orbit_list(self, scmatrix):
        """
        Generate list of data to compute correlation vectors.

        List includes orbit bit ids, flat correlation tensors and their indices,
        and array of cluster indices for supercell corresponding to given
        supercell matrix.

        This is a helper function for the correlation vector computation and most
        often called internally.

        Args:
            scmatrix (ndarray):
                supercell matrix.

        Returns: list of tuples
            [(orbit bit ids, tensor  indices, flat corr tensors, cluster indices)]
        """
        mappings = self.supercell_orbit_mappings(scmatrix)
        orbit_list = []
        for orbit, cluster_inds in zip(self.orbits, mappings):
            orbit_list.append(
                (
                    orbit.bit_id,
                    orbit.flat_tensor_indices,
                    orbit.flat_correlation_tensors,
                    cluster_inds,
                )
            )

        return orbit_list

    def _assign_orbit_ids(self):
        """Assign unique id's to orbit.

        Assign unique id's to each orbit based on all its orbit functions and
        all clusters in the prim structure that are in each orbit.
        """
        counts = (1, 1, 1)
        for key in sorted(self._orbits.keys()):
            for orbit in self._orbits[key]:
                counts = orbit.assign_ids(*counts)

        self.num_orbits = counts[0]
        self.num_corr_functions = counts[1]
        self.num_clusters = counts[2]

    def get_orbit_indices(self, scmatrix):
        """Get the OrbitIndices named tuple for a given supercell matrix.

        If the indices have not been cached then they are generated by generating
        the site mappings for the given supercell.e
        """
        # np.arrays are not hashable and can't be used as dict keys.
        scmatrix = np.array(scmatrix)
        scm = tuple(sorted(tuple(s.tolist()) for s in scmatrix))
        orbit_indices = self._supercell_orbit_inds.get(scm)

        if orbit_indices is None:
            orbit_indices = self._gen_orbit_indices(scmatrix)
            self._supercell_orbit_inds[scm] = orbit_indices

        return orbit_indices

    def _gen_orbit_indices(self, scmatrix):
        """Find all the cluster site indices associated with each orbit in structure.

        The structure corresponding to the given supercell matrix w.r.t prim.
        """
        supercell = self.structure.copy()
        supercell.make_supercell(scmatrix)
        prim_to_supercell = np.linalg.inv(scmatrix)
        supercell_fcoords = np.array(supercell.frac_coords)

        pts = lattice_points_in_supercell(scmatrix)
        orbit_indices = []
        for orbit in self.orbits:
            prim_fcoords = np.array([c.frac_coords for c in orbit.clusters])
            fcoords = np.dot(prim_fcoords, prim_to_supercell)
            # tcoords contains all the coordinates of the symmetrically
            # equivalent clusters the indices are: [equivalent cluster
            # (primitive cell), translational image, index of site in cluster,
            # coordinate index]
            tcoords = fcoords[:, None, :, :] + pts[None, :, None, :]
            tcs = tcoords.shape
            inds = coord_list_mapping_pbc(
                tcoords.reshape((-1, 3)), supercell_fcoords, atol=SITE_TOL
            ).reshape((tcs[0] * tcs[1], tcs[2]))
            # orbit_ids holds orbit, and 2d array of index groups that
            # correspond to the orbit
            # the 2d array may have some duplicates. This is due to
            # symmetrically equivalent groups being matched to the same sites
            # (eg in simply cubic all 6 nn interactions will all be [0, 0]
            # indices. This multiplicity disappears as supercell_structure size
            # increases, so I haven't implemented a more efficient method

            # assure contiguous C order
            orbit_indices.append(np.ascontiguousarray(inds, dtype=int))

        orbit_indices = tuple(orbit_indices)
        return OrbitIndices(orbit_indices, IntArray2DContainer(orbit_indices))

    @staticmethod
    def _gen_orbits_from_cutoffs(
        exp_struct, cutoffs, symops, basis, orthonorm, use_conc
    ):
        """Generate orbits from diameter cutoffs.

        The diameter of a cluster is the maximum distance between any two
        sites in the cluster.

        Generates dictionary of {size: [Orbits]} given a dictionary of maximal
        cluster diameters and symmetry operations to apply (not necessarily all
        the symmetries of the expansion_structure).

        Args:
            exp_struct (Structure):
                Structure with all sites that have partial occupancy.
            cutoffs (dict):
                dict of cutoffs for cluster diameters {size: cutoff}.
                Cutoff diameters must decrease with cluster size, otherwise
                algorithm can not guarantee completeness of cluster subspace.
                Adding a cutoff term for point terms, ie {1: 0} is useful
                to exclude point terms any other value for the cutoff will
                simply be ignored.
            symops (list of SymmOps):
                list of symmetry operations for structure
            basis (str):
                name identifying site basis set to use.
            orthonorm (bool):
                whether to ensure orthonormal basis set.
            use_conc (bool):
                If true the concentrations in the prim structure sites will be
                used as the measure to orthormalize site bases.
        Returns:
            dict: {size: list of Orbits within diameter cutoff}
        """
        try:
            if cutoffs.pop(1) is None:
                if len(cutoffs) != 0:
                    raise ValueError(
                        f"Unable to generate clusters of higher order "
                        f" {cutoffs} if point terms are excluded."
                    )
                return {}
        except KeyError:
            pass

        site_spaces = get_site_spaces(exp_struct, include_measure=use_conc)
        site_bases = tuple(
            basis_factory(basis, site_space) for site_space in site_spaces
        )
        if orthonorm:
            for s_basis in site_bases:
                s_basis.orthonormalize()

        orbits = {}
        nbits = np.array([len(b) - 1 for b in site_spaces])

        # Generate singlet/point orbits
        orbits[1] = ClusterSubspace._gen_point_orbits(
            exp_struct, site_bases, nbits, symops
        )

        if len(cutoffs) == 0:  # return singlets only if no cutoffs provided
            return orbits

        orbits.update(
            ClusterSubspace._gen_multi_orbits(
                orbits[1], exp_struct, cutoffs, site_bases, nbits, symops
            )
        )
        return orbits

    @staticmethod
    def _gen_point_orbits(exp_struct, site_bases, nbits, symops):
        """Generate point orbits.

        Args:
            nbits (ndarray):
                array with total values for function indices per site.
            exp_struct (Structure):
                expansion structure, disordered sites only.
            site_bases (list of DiscreteBasis):
                list of site basis for each site in the expansion structure.
            symops (list of SymmOp):
                lists of symmetry operations of the underlying structure.

        Returns:
            list of Orbits:
                list of point orbits.
        """
        pt_orbits = []
        for nbit, site, sbasis in zip(nbits, exp_struct, site_bases):
            # Coordinates of point terms must stay in [0, 1] to guarantee
            # correct math of the following algorithm.
            new_orbit = Orbit(
                [np.mod(site.frac_coords, 1)],
                exp_struct.lattice,
                [list(range(nbit))],
                [sbasis],
                symops,
            )
            if new_orbit not in pt_orbits:
                pt_orbits.append(new_orbit)

        # sorted by decreasing crystallographic multiplicity and finally by increasing
        # number of correlation functions (bit combos) -> so that higher symmetry orbits
        # come first
        pt_orbits = sorted(
            pt_orbits,
            key=lambda x: (
                -x.multiplicity,
                len(x),
            ),
        )
        return pt_orbits

    @staticmethod
    def _gen_multi_orbits(point_orbits, exp_struct, cutoffs, site_bases, nbits, symops):
        """Generate point orbits.

        Args:
            point_orbits (list of Orbit):
                list of point orbits.
            exp_struct (Structure):
                expansion structure, disordered sites only.
            cutoffs (dict):
                dict of cutoffs for cluster diameters {size: cutoff}.
                Cutoff diameters must decrease with cluster size, otherwise
                algorithm can not guarantee completeness of cluster subspace.
                Adding a cutoff term for point terms, ie {1: None} is useful
                to exclude point terms any other value for the cutoff will
                simply be ignored.
            site_bases (list of DiscreteBasis):
                list of site basis for each site in the expansion structure.
            nbits (ndarray):
                array with total values for function indices per site.
            symops (list of SymmOp):
                lists of symmetry operations of the underlying structure.

        Returns:
            dict:
                {size: list of Orbits within diameter cutoff}
        """
        # diameter + max_lp gives maximum possible distance from
        # [0.5, 0.5, 0.5] prim centoid to a point in all enumerable
        # clusters. Add SITE_TOL as a numerical tolerance grace.
        orbits = {1: point_orbits}
        centroid = exp_struct.lattice.get_cartesian_coords([0.5, 0.5, 0.5])
        coords = exp_struct.lattice.get_cartesian_coords(exp_struct.frac_coords)
        max_lp = max(np.sum((coords - centroid) ** 2, axis=-1) ** 0.5) + SITE_TOL

        for size, diameter in sorted(cutoffs.items()):
            new_orbits = []
            neighbors = exp_struct.get_sites_in_sphere(
                centroid, diameter + max_lp, include_index=True
            )
            for orbit in orbits[size - 1]:
                if orbit.base_cluster.diameter > diameter:
                    continue

                for neighbor in neighbors:
                    if is_coord_subset(
                        [neighbor.frac_coords],
                        orbit.base_cluster.frac_coords,
                        atol=SITE_TOL,
                    ):
                        continue

                    new_sites = np.concatenate(
                        [orbit.base_cluster.frac_coords, [neighbor.frac_coords]]
                    )
                    new_orbit = Orbit(
                        new_sites,
                        exp_struct.lattice,
                        orbit.bits + [list(range(nbits[neighbor.index]))],
                        orbit.site_bases + [site_bases[neighbor.index]],
                        symops,
                    )

                    if new_orbit.base_cluster.diameter > diameter + 1e-8:
                        continue

                    if new_orbit not in new_orbits:
                        new_orbits.append(new_orbit)

            # sorted by increasing cluster diameter, then by decreasing crystallographic
            # multiplicity and finally by increasing number of correlation functions
            # (bit combos) -> so that higher symmetry orbits come first
            if len(new_orbits) > 0:
                orbits[size] = sorted(
                    new_orbits,
                    key=lambda x: (
                        np.round(x.base_cluster.diameter, 6),
                        -x.multiplicity,
                        len(x),
                    ),
                )
        return orbits

    def __eq__(self, other):
        """Check equality between cluster subspaces."""
        if not isinstance(other, ClusterSubspace):
            return False
        if other.num_corr_functions != self.num_corr_functions:
            return False
        if len(self.external_terms) != len(other.external_terms):
            return False
        if not all(
            isinstance(t1, type(t2))
            for t1, t2 in zip(other.external_terms, self.external_terms)
        ):
            return False
        # does not check if basis functions are the same.
        return all(o1 == o2 for o1, o2 in zip(other.orbits, self.orbits))

    def __contains__(self, orbit):
        """Check if subspace contains orbit."""
        return orbit in self.orbits

    def __len__(self):
        """Get number of correlation functions and ext terms in subspace."""
        return self.num_corr_functions + len(self.external_terms)

    def __str__(self):
        """Convert class into pretty string for printing."""
        outs = [
            f"Basis/Orthogonal/Orthonormal : {self.basis_type}/{self.basis_orthogonal}/"
            f"{self.basis_orthonormal}",
            f"       Unit Cell Composition : {self.structure.composition}",
            f"            Number of Orbits : {self.num_orbits}",
            f"No. of Correlation Functions : {self.num_corr_functions}",
            "             Cluster Cutoffs : "
            f"{', '.join('{}: {:.2f}'.format(s, c) for s, c in self.cutoffs.items())}",
            f"              External Terms : {self.external_terms}",
            "Orbit Summary",
            " ------------------------------------------------------------------------",
            " |  ID     Degree    Cluster Diameter    Multiplicity    No. Functions  |",
            " |   0       0             NA                 0                1        |",
        ]
        for degree, orbits in self.orbits_by_size.items():
            for orbit in orbits:
                outs.append(
                    f" |{orbit.id:^7}{degree:^10}{orbit.base_cluster.diameter:^20.4f}"
                    f"{orbit.multiplicity:^16}{len(orbit):^17}|"
                )
        outs.append(
            " ------------------------------------------------------------------------"
        )
        return "\n".join(outs)

    def __repr__(self):
        """Return a summary of subspace."""
        outs = [
            "Cluster Subspace Summary",
            f"Basis/Orthogonal/Orthonormal : {self.basis_type}/{self.basis_orthogonal}/"
            f"{self.basis_orthonormal}",
            f"Unit Cell Composition : {self.structure.composition}",
            f"Number of Orbits : {self.num_orbits}   "
            f"No. of Correlation Functions : {self.num_corr_functions}",
            "Cluster Cutoffs : "
            f"{', '.join('{}: {:.2f}'.format(s, c) for s, c in self.cutoffs.items())}",
            f"External Terms : {self.external_terms}",
        ]
        return "\n".join(outs)

    @classmethod
    def from_dict(cls, d):
        """Create ClusterSubspace from an MSONable dict."""
        symops = [SymmOp.from_dict(so_d) for so_d in d["symops"]]
        orbits = {
            int(s): [Orbit.from_dict(o) for o in v] for s, v in d["orbits"].items()
        }
        structure = Structure.from_dict(d["structure"])
        exp_structure = Structure.from_dict(d["expansion_structure"])
        sc_matcher = StructureMatcher.from_dict(d["sc_matcher"])
        site_matcher = StructureMatcher.from_dict(d["site_matcher"])
        cluster_subspace = cls(
            structure=structure,
            expansion_structure=exp_structure,
            orbits=orbits,
            symops=symops,
            supercell_matcher=sc_matcher,
            site_matcher=site_matcher,
        )

        # attempt to recreate external terms. This can be much improved if
        # a base class is used.
        # TODO update this using instances of BasePairTerm when the time comes
        for term in d["external_terms"]:
            try:
                module = import_module(term["@module"])
                term_class = getattr(module, term["@class"])
                cluster_subspace.add_external_term(term_class.from_dict(term))
            except AttributeError:
                warnings.warn(
                    f"{term['@class']} was not found in {term['@module']}. You"
                    f" will need to add this yourself.",
                    RuntimeWarning,
                )
            except ImportError:
                warnings.warn(
                    f"Module {term['@module']} for class {term['@class']} was "
                    f"not found. You will have to add this yourself.",
                    ImportWarning,
                )
        # re-create supercell orb inds cache
        _supercell_orbit_inds = {}
        for scm, indices in d["_supercell_orb_inds"]:
            scm = tuple(tuple(s) for s in scm)
            if isinstance(indices[0][0], int) and isinstance(indices[0][1], list):
                warnings.warn(
                    "This ClusterSubspace was created with a previous version "
                    "of smol. Please resave it to avoid this warning.",
                    FutureWarning,
                )
                _supercell_orbit_inds[scm] = tuple(
                    np.array(ind) for o_id, ind in indices
                )
            else:
                _supercell_orbit_inds[scm] = tuple(np.array(ind) for ind in indices)
        # now generate the containers
        _supercell_orbit_inds = {
            scm: OrbitIndices(indices, IntArray2DContainer(indices))
            for scm, indices in _supercell_orbit_inds.items()
        }
        cluster_subspace._supercell_orbit_inds = _supercell_orbit_inds
        return cluster_subspace

    def as_dict(self):
        """
        Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        # modify cached sc orbit indices so it can be serialized
        _supercell_orb_inds = [
            (scm, [indices.tolist() for indices in orbit_inds.arrays])
            for scm, orbit_inds in self._supercell_orbit_inds.items()
        ]

        cs_dict = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "structure": self.structure.as_dict(),
            "expansion_structure": self.expansion_structure.as_dict(),
            "sc_matcher": self._sc_matcher.as_dict(),
            "site_matcher": self._site_matcher.as_dict(),
            "symops": jsanitize(self.symops, strict=True),
            "orbits": jsanitize(self._orbits, strict=True),
            "external_terms": jsanitize(self.external_terms, strict=True),
            "_supercell_orb_inds": _supercell_orb_inds,
        }
        return cs_dict


def invert_mapping(mapping):
    """Invert a mapping table from forward to backward, vice versa.

    Args:
        mapping (list of lists):
            List of sublists, each contains integer indices, indicating
            a forward mapping from the current sublist index to the indices
            in the sublist.

    Returns:
        lists of lists: Inverted mapping table containing backward mapping.
        Same format as input.
    """
    inv_mapping = [[] for _ in range(len(mapping))]

    for i in range(len(mapping) - 1, -1, -1):
        values_list = mapping[i]
        for value in values_list:
            if i not in inv_mapping[value]:
                inv_mapping[value].append(i)

    return inv_mapping


def get_complete_mapping(mapping):
    """Get a complete mapping from a 1 level mapping.

    If we allow transferability between mapping linkages, there would be
    a full mapping containing all linkages at all connectivity level.
    Using this function, you can get a full mapping from an incomplete
    mapping.

    NOTE: Since complete hierarchy is not very useful for actual CE fit, we
    will not include it as an attribute of ClusterSubspace.

    Args:
        mapping (list of lists):
             List of sublists, each contains integer indices, indicating
            a forward mapping from the current sublist index to the indices
            in the sublist.

    Returns:
        list of lists: Full mapping table containing forward mapping, but with
        all connectivity levels. Same format as input
    """
    all_level_mapping = deepcopy(mapping)

    for i in range(len(mapping) - 1, -1, -1):
        next_values_list = mapping[i]

        while len(next_values_list) > 0:
            for next_value in next_values_list:
                if next_value not in all_level_mapping[i]:
                    all_level_mapping[i].append(next_value)
            next_values_list_new = []
            for next_value in next_values_list:
                for nn_value in mapping[next_value]:
                    if nn_value not in next_values_list_new:
                        next_values_list_new.append(nn_value)
            next_values_list = next_values_list_new

    return all_level_mapping


class PottsSubspace(ClusterSubspace):
    """PottsSubspace represents a subspace of functions using only indicator functions.

    A PottsSubspace is a related model to a standard ClusterSubspace used to build
    a standard cluster expansion. The only difference is that the single site functions
    for any orbit are only made up of indicator functions (and there is no constant
    function). As such it is more closely related to a generalized Ising model or better
    yet an extension of the q-state Potts Model (hence the name).

    The orbit functions in a PottsSubspace represent the concentrations of all possible
    decorations (species occupancies) of clusters in the given orbit. Similar to a
    ClusterSubspace with site indicator basis functions. But in contrast, the Potts
    subspace includes the concentration of all possible decorations (minus 1).

    Although quite similar to a ClusterSubspace, there is no mathematical
    formalism guaranteeing that the orbit basis functions generated in a
    PottsSubspace are a linear independent set spanning configuration space.
    Actually if all orbits up to any size (or infinite size) are included,
    the corresponding orbit function set is an overcomplete/ highly redundant.

    A PottsSubspace can be created directly with a ClusterSubspace object
    by using the constructor and providing the appropriately constructed orbits,
    but that is a bit more painful that just using the class method here.
    """

    def __init__(
        self,
        structure,
        expansion_structure,
        symops,
        orbits,
        without_last_cluster=True,
        supercell_matcher=None,
        site_matcher=None,
        **matcher_kwargs,
    ):
        """Initialize a PottsSubspace.

        You rarely will need to create a ClusterSubspace using the main
        constructor.
        Look at the class method :code:`from_cutoffs` for the "better" way to
        instantiate a ClusterSubspace.

        Args:
            structure (Structure):
                Structure to define the cluster space. Typically the primitive
                cell. Includes all species regardless of partial occupation.
            expansion_structure (Structure):
                Structure including only sites that will be included in the
                Cluster space. (only those with partial occupancy)
            symops (list of Symmop):
                list of Symmops for the given structure.
            orbits (dict): {size: list of Orbits}
                Dictionary with size (number of sites) as keys and list of
                Orbits as values.
            without_last_cluster (bool): optional
                whether last cluster labeling is removed from each orbit.
            supercell_matcher (StructureMatcher): (optional)
                StructureMatcher used to find supercell matrices
                relating the prim structure to other structures. If you pass
                this directly you should know how to set the matcher up,
                otherwise matching your relaxed structures can fail, a lot.
            site_matcher (StructureMatcher): (optional)
                StructureMatcher used to find site mappings
                relating the sites of a given structure to an appropriate
                supercell of the prim structure . If you pass this directly you
                should know how to set the matcher up, otherwise matching your
                relaxed structures can fail, a lot.
            matcher_kwargs:
                ltol, stol, angle_tol, supercell_size: parameters to pass
                through to the StructureMatchers. Structures that don't match
                to the primitive cell under these tolerances won't be included
                in the expansion. Easiest option for supercell_size is usually
                to use a species that has a constant amount per formula unit.
                See pymatgen documentation of :class:`StructureMatcher` for
                more details.
        """
        self._wo_last_cluster = without_last_cluster
        super().__init__(
            structure,
            expansion_structure,
            symops,
            orbits,
            supercell_matcher,
            site_matcher,
            **matcher_kwargs,
        )

    @classmethod
    def from_cutoffs(
        cls,
        structure,
        cutoffs,
        remove_last_cluster=False,
        supercell_matcher=None,
        site_matcher=None,
        **matcher_kwargs,
    ):
        """Create a PottsSubspace from diameter cutoffs.

        Creates a :class:`PottsSubspace` with orbits of the given size and
        diameter smaller than or equal to the given value. The diameter of an
        orbit is the maximum distance between any two sites of a cluster of
        that orbit.

        Args:
           structure (Structure):
               disordered structure to build a cluster expansion for.
               Typically the primitive cell
           cutoffs (dict):
               dict of {cluster_size: diameter cutoff}. Cutoffs should be
               strictly decreasing. Typically something like {2:5, 3:4}.
               The empty orbit is always included. Singlets are by default
               included, with the exception below.
               To obtain a subspace with only an empty and singlet terms use
               an empty dict {}, or {1: 1}. Adding a cutoff term for point
               terms, i.e. {1: None} is useful to exclude point terms, any other
               value for the cutoff will simply be ignored.
           remove_last_cluster (bool): optional
               if True, will remove the last cluster labeling (decoration)
               from each orbit. Since sum of corr for all labelings = 1,
               removing the last is similar to working in concentration space.
           supercell_matcher (StructureMatcher): (optional)
               StructureMatcher used to find supercell matrices
               relating the prim structure to other structures. If you pass
               this directly you should know how to set the matcher up,
               otherwise matching your relaxed structures will fail, a lot.
           site_matcher (StructureMatcher): (optional)
               StructureMatcher used to find site mappings
               relating the sites of a given structure to an appropriate
               supercell of the prim structure . If you pass this directly you
               should know how to set the matcher up, otherwise matching your
               relaxed structures will fail, a lot.
           matcher_kwargs:
               ltol, stol, angle_tol, supercell_size: parameters to pass
               through to the StructureMatchers. Structures that don't match
               to the primitive cell under these tolerances won't be included
               in the expansion. Easiest option for supercell_size is usually
               to use a species that has a constant amount per formula unit.

        Returns:
           PottsSubSpace
        """
        # get symmetry operations of prim structure.
        symops = SpacegroupAnalyzer(structure).get_symmetry_operations()
        # get the active sites (partial occupancy) to expand over.
        sites_to_expand = [
            site
            for site in structure
            if site.species.num_atoms < 0.99 or len(site.species) > 1
        ]
        expansion_structure = Structure.from_sites(sites_to_expand)
        # get orbits within given cutoffs
        orbits = cls._gen_orbits_from_cutoffs(
            expansion_structure, cutoffs, symops, remove_last_cluster
        )
        return cls(
            structure=structure,
            expansion_structure=expansion_structure,
            symops=symops,
            orbits=orbits,
            without_last_cluster=remove_last_cluster,
            supercell_matcher=supercell_matcher,
            site_matcher=site_matcher,
            **matcher_kwargs,
        )

    def get_function_decoration(self, index):
        """Get the decoration/labeling of a specific orbit function.

        When using an indicator site basis there is 1 to 1 equivalence between
        correlation functions and species decorations.

        Args:
            index (int):
                index of orbit function in correlation vector

        Returns:
            list of tuples: list of tuples of symmetrically equivalent
            Species/Elements.
        """
        orbit = self.orbits[self.function_orbit_ids[index] - 1]
        decorations = [
            tuple(list(orbit.site_spaces[i])[b] for i, b in enumerate(bits))
            for bits in orbit.bit_combos[index - orbit.bit_id]
        ]
        return decorations

    def get_orbit_decorations(self, orbit_id):
        """Get all decorations/labellings of species in an orbit.

        Args:
            orbit_id (int):
                ID of orbit

        Returns:
            list of list: list of lists of symmetrically equivalent
            Species/Elements.
        """
        bit_id = self.orbits[orbit_id - 1].bit_id
        num_combos = len(self.orbits[orbit_id - 1].bit_combos)
        return [
            self.get_function_decoration(bid)
            for bid in range(bit_id, bit_id + num_combos)
        ]

    @staticmethod
    def _gen_orbits_from_cutoffs(exp_struct, cutoffs, symops, remove_last):
        """Generate orbits from diameter cutoffs.

        Generates dictionary of orbits in the same way that the cluster
        subspace class does, except that the orbit functions (and corresponding
        bit combos) include all symmetrically distinct decorations/labelings
        of indicator functions for all allowed species (except 1 decoration
        for each orbit since this value is just 1 - sum of concentration of all
        other decorations

        Args:
            exp_struct (Structure):
                Structure with all sites that have partial occupancy.
            cutoffs (dict):
                dict of cutoffs for cluster diameters {size: cutoff}
            symops (list of SymmOps):
                list of symmetry operations for structure
            remove_last (bool):
                remove the last cluster labeling from each orbit.

        Returns:
            dict: {size: list of Orbits within diameter cutoff}
        """
        site_spaces = get_site_spaces(exp_struct)
        site_bases = tuple(IndicatorBasis(site_space) for site_space in site_spaces)
        orbits = {}
        nbits = np.array([len(b) for b in site_spaces])

        try:
            if cutoffs.pop(1) is None:
                if len(cutoffs) != 0:
                    raise ValueError(
                        f"Unable to generate clusters of higher order "
                        f" {cutoffs} if point terms are excluded."
                    )
                return {}
        except KeyError:
            pass

        # Generate singlet/point orbits
        orbits[1] = ClusterSubspace._gen_point_orbits(
            exp_struct, site_bases, nbits, symops
        )

        if len(cutoffs) == 0:  # return singlets only if no cutoffs provided
            return orbits

        orbits.update(
            ClusterSubspace._gen_multi_orbits(
                orbits[1], exp_struct, cutoffs, site_bases, nbits, symops
            )
        )

        if remove_last:
            for orbs in orbits.values():
                for orb in orbs:
                    orb.remove_bit_combos_by_inds([len(orb.bit_combos) - 1])

        return orbits

    def as_dict(self):
        """
        Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        # pylint: disable=protected-access, no-member
        pcs_dict = super().as_dict()
        pcs_dict["_wo_last_cluster"] = self._wo_last_cluster
        return pcs_dict

    @classmethod
    def from_dict(cls, d):
        """Create ClusterSubspace from an MSONable dict."""
        # pylint: disable=protected-access, no-member
        subspace = super().from_dict(d)
        subspace._wo_last_cluster = d.get("_wo_last_cluster", True)
        # remove last bit combo in all orbits
        if subspace._wo_last_cluster:
            for orbit in subspace.orbits:
                orbit.remove_bit_combos_by_inds([len(orbit.bit_combos) - 1])
        subspace._assign_orbit_ids()
        return subspace
