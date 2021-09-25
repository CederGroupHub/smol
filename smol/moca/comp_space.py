"""Compositional space file.

This file contains functions related to implementing and navigating the
compositional space.

In CEAuto, we first define a starting, charge-neutral, and fully occupied
('Vac' is also an occupation) composition in the compositional space.

Then we find all possible unitary, charge and number conserving flipping
combinations of species, and define them as axis in the compositional space.

A composition is then defined as a vector, with each component corresponding
to the number of filps that have to be done on a 'flip axis' in order to get
the target composition from the defined, starting composition.

For some supercell size, a composition might not be 'reachable' because
supercell_size*atomic_ratioo is not an integer. For this case, you need to
select a proper enumeration fold for your compositions(see enum_utils.py),
or choose a proper supercell size.

Case test in [[Li+,Mn3+,Ti4+],[P3-,O2-]] passed.
"""


__author__ = "Fengyu Xie"


import numpy as np
import polytope as pc
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

from itertools import product, chain
from copy import deepcopy
from monty.json import MSONable, MontyDecoder

from pymatgen.core import Element, Composition
from smol.cofe.space.domain import Vacancy

from .utils.math_utils import (get_integer_basis, integerize_multiple,
                               get_integer_base_solution,
                               get_integer_grid)


NUMCONERROR = ValueError("Operation error, flipping not number \
                         conserved.")
OUTOFSUBLATERROR = ValueError("Operation error, flipping between \
                              different sublattices.")
CHGBALANCEERROR = ValueError("Charge balance cannot be achieved \
                             with these species.")
OUTOFSUBSPACEERROR = ValueError("Given coordinate falls outside \
                                the subspace.")
SLACK_TOL = 1E-5


# Finding minimun charge-conserved, number-conserved flips to establish
# constrained coords system.
def get_oxi_state(sp):
    """Oxidation state from Specie/Element/Vacancy.

    Args:
       sp(Specie):
          A species object.
    Return:
       Int.
    """
    if isinstance(sp, (Vacancy, Element)):
        return 0
    else:
        return sp.oxi_state


def get_unit_excitations(bits):
    """One atom excitations.

    Get all possible single site flips on each sublattice, and the
    charge changes given by the flips. Flips will be encoded by
    species indices, and will always flip from the last specie in
    the site space, to another specie on the same sublattice.

    Args:
        bits(List[List[Speice|Element|Vacancy]]):
            a list of Species or DummySpecies on each sublattice.
            For example:
            [[Specie.from_string('Ca2+'),Specie.from_string('Mg2+'))],
             [Specie.from_string('O2-')]]
    Return:
        tuple:(list of flips in unconstrained space,
               list of charge changes of each flip,
               list of flip indices in each sublattice)
    """
    unit_excitations = []
    unit_n_excitations = []
    excitation_ids_by_sublat = []
    cur_exi_id = 0
    for sl_id, sl_sps in enumerate(bits):
        unit_excitations.extend([(sp, sl_sps[-1], sl_id)
                                for sp in sl_sps[:-1]])
        unit_n_excitations.extend([(sp_id, len(sl_sps) - 1, sl_id)
                                  for sp_id in range(len(sl_sps) - 1)])
        excitation_ids_by_sublat.append([cur_exi_id + i for i in
                                        range(len(sl_sps) - 1)])
        cur_exi_id += (len(sl_sps) - 1)

    chg_of_excitations = [int(get_oxi_state(p[0]) - get_oxi_state(p[1]))
                          for p in unit_excitations]

    return unit_n_excitations, chg_of_excitations, excitation_ids_by_sublat


def flip_vecs_to_flip_table(unit_n_excitations, n_bits, flip_vecs):
    """Turn multi-atom flip vector to flip table format.

    Translates flips from their vector from into their
    dictionary form.
    Each dictionary is written in the form below:
    {
     'from':
           {sublattice index:
               {specie index in site space:
                   number_of_this_specie_to_be_removed_from_the_sublat
               }
                ...
           }
           ...
     'to':
           {
           ...     number_of_this_specie_to_be_generated_on_this_sublat
           }
    }
    Args:
        unit_n_excitations(List[Tuple(int)]):
            A flatten list of all possible, single site flips
            represented in specie indices, each term written as:
             (flip_to_specie_index,flip_from_specie_index,
              sublattice_index_of_flip)
        n_bits(List[List[int]]):
            List containing specie indices in on each sublattice.
        flip_vecs(2D ArrayLike):
            Vectors in the charge neutral space, can be interpreted as
            charge neutral flips.
    Return:
        List[Dict]
    """
    n_sls = len(n_bits)
    flip_table = []

    for flip_vec in flip_vecs:
        flip_dict = {'from': {}, 'to': {}}

        flip_dict['from'] = {sl_id: {sp_id: 0 for sp_id in n_bits[sl_id]}
                             for sl_id in range(n_sls)}
        flip_dict['to'] = {sl_id: {sp_id: 0 for sp_id in n_bits[sl_id]}
                           for sl_id in range(n_sls)}

        for flip, n_flip in zip(unit_n_excitations, flip_vec):
            if n_flip > 0:
                flip_to, flip_from, sl_id = flip
                n = n_flip
            elif n_flip < 0:
                flip_from, flip_to, sl_id = flip
                n = -1 * n_flip
            else:
                continue

            flip_dict['from'][sl_id][flip_from] += n
            flip_dict['to'][sl_id][flip_to] += n

        # Simplify ionic equations
        flip_dict_clean = {'from': {}, 'to': {}}
        for sl_id in range(n_sls):
            for sp_id in n_bits[sl_id]:
                del_n = (flip_dict['from'][sl_id][sp_id]
                         - flip_dict['to'][sl_id][sp_id])

                if del_n > 0:
                    if sl_id not in flip_dict_clean['from']:
                        flip_dict_clean['from'][sl_id] = {}
                    flip_dict_clean['from'][sl_id][sp_id] = del_n
                elif del_n < 0:
                    if sl_id not in flip_dict_clean['to']:
                        flip_dict_clean['to'][sl_id] = {}
                    flip_dict_clean['to'][sl_id][sp_id] = -1 * del_n
                else:
                    continue

        flip_table.append(flip_dict_clean)

    return flip_table


def flip_table_to_flip_vecs(n_bits, flip_table):
    """Compute flip vectors (unconstrained coords) from flip table.

    Args:
        n_bits(List[List[int]]):
            List containing indices of species on sublattices.
        flip_table(Dict):
            Flip table, same as format described in documentation of
            flip_vecs_to_flip_table.
    Returns:
        np.ndarray: Flip vectors in unconstrained coordinates.
    """
    flip_vecs = []
    for flip in flip_table:
        dcompstat = [[0 for sp_id in sl] for sl in n_bits]
        for sl_id in flip['from']:
            for sp_id in flip["from"][sl_id]:
                dcompstat[sl_id][sp_id] -= flip['from'][sl_id][sp_id]
        for sl_id in flip['to']:
            for sp_id in flip["to"][sl_id]:
                dcompstat[sl_id][sp_id] += flip['to'][sl_id][sp_id]
        # Assert number conserving.
        assert sum(chain(*dcompstat)) == 0
        v = []
        for dsl in dcompstat:
            v.extend(dsl[:-1])
        flip_vecs.append(v)

    return np.array(flip_vecs, dtype=np.int64)


def visualize_flip_table(flip_table, bits):
    """Turn multi-atom flip table into reaction formulas.

    Turns a charge constrained flip from dictionary format
    into a string of reaction formula for easy interpretation.
    Args:
        flip_table(List[dict]):
            List of flips in dictionary format. See doc of flipvec_to_
            operation for details.
        bits(List[List[Specie|DummySpecie|Element|Vacancy]]):
            A list containing a species on all sublattices.
    Return:
        List[str].
    """
    formula_strs = []

    for flip_dict in flip_table:
        from_strs = []
        to_strs = []

        for sl_id in flip_dict['from']:
            for flip_from, n in flip_dict['from'][sl_id].items():
                from_name = str(bits[sl_id][flip_from])
                from_strs.append('{} {}({})'
                                 .format(n, from_name, sl_id))

        for sl_id in flip_dict['to']:
            for flip_to, n in flip_dict['to'][sl_id].items():
                to_name = str(bits[sl_id][flip_to])
                to_strs.append('{} {}({})'
                               .format(n, to_name, sl_id))

        from_str = ' + '.join(from_strs)
        to_str = ' + '.join(to_strs)
        formula_strs.append(from_str + ' -> ' + to_str)

    return formula_strs


# Compsitional space class
class CompSpace(MSONable):
    """Compositional space class.

    Generates a charge neutral compositional space from a list of Species
    or DummySpecies and size of sublattices in a PRIMITIVE CELL.

    A composition in CEAuto can be expressed in 5 forms:
    1, A Coordinate in compositional space without the charge constraint,
       which uses one site flips from every last specie on sublattices
       to every other species on the same sublattice as basis vectors.
       An imaginary occupation with all sublattice sites occupied by the
       last specie is defined as a background occupation, and the coordinates
       are defined as the number of each unconstrained flips (as above)
       required to reach a certain composition from the background occupation.
       Define this as unconstrained coordinate. 'unconstr_coord'.

    2, A Coordinate in constrained, charge neutral subspace of the
       unconstrained space, with charge neutral, number
       conserving elementary flips (usually combined, manybody flips)
       as basis vectors, and a selected charge neutral composition as
       its origin.
       (Usually selected as one vertex of the constrained space.)
       We call this constrained coordinate ('constr_coord').

    3, List of species counts on each sublattices, in the order of the
       SiteSpace species table.
       We call this compositional_statistics ('compstat').

    4, Normalized compostions of each sublattice.('composition').Vacancies
       will not be explicitly included as keys.

       Note: This form is always normalized by sublattice sizes,
             while other form will only be normalized by the supercell size.

    5, The above are all discriminate the same specie on different sublattice
       as different coordinate components.
       By summing the count of same specie on different sublattices into one
       coordinate, we can establish the non-discriminative coordinates
       ('nondisc').

    For example, if bits = [[Li+,Mn3+,Ti4+],[P3-,O2-]] and sl_sizes = [1,1]
    (LMTOF rock-salt), then:
       Unconstrained space basis are:
            Ti4+ -> Li+, Ti4+ -> Mn3+, O2- -> P3-

       Back ground occupation shall be:
            (1 Ti4+ |1 O-),supercell size = 1
       The dimensionality of the unconstrained space is 3.

       Charge-neutral combined flips basis shall be:
            3 Mn3+ -> 2 Ti4+ + Li+, Ti4+ + P3- -> O2- + Mn3+

       Charge neutral compositional space origin can be chosen as:
            (Mn3+ | P-),supercell size = 1
        The constrained subspace's dimensionality is 2.

    Given composition:
        (Li0.5 Mn0.5| O), supercell size=1
    It's unconstrained coordinates will be (0.5,0.5,0), and the constrained
    coordinates will be (0.5,1.0).

    When the system is not charged (all the flips are charge conserved, and
    the background occupation has zero charge), then the constrained
    space will be the same as the unconstrained space.

    Compspace class provides methods for you to convert between all these
    representations easily.
    It will also allow you to enumerate all possible integer compositions in a
    given supercell size(shall be used as a composition enumeration in CEAuto).
    All enumerations are done with integer enumeration methods.

    Again, even if all other coordinates are in integer form, the converted
    'composition' form will always be normalized.

    Attributes:
        species(List[Species|DummySpecies|Element|Vacancy]):
            All species in the given system (sorted).
        n_bits(List[List[int]]):
            Species indices on each sublattice, used to decode
            or encode occupations and the dictionary flips format.
        dim(int):
            Dimensionality of the charge constrained space.
        min_flip_table(List[Dict]):
            All minimal, charge neutral and number conserving flip
            combinations in the given system, in dictionary format.
    """

    def __init__(self, bits, sl_sizes=None):
        """Initialize CompSpace.

        Args:
            bits(List of Specie/DummySpecie):
                bit list.
                Sorted before use. We don't sort it here in case the order
                of Vacancy is broken, so be careful.
            sl_sizes(List[int]):
                Sublattice sizes in a PRIMITIVE cell.
                If None given, sl_sizes will be reset to [1,1,....]
        """
        self.bits = bits

        # For non-discriminative coordinates
        species = list(set(chain(*self.bits)))
        self.species = []
        for b in species:
            if isinstance(b, Vacancy):
                vac_dupe = False
                for b_old in self.species:
                    if isinstance(b_old, Vacancy):
                        vac_dupe = True
                        break
                if vac_dupe:
                    continue
            self.species.append(b)

        # Sort the species
        self.species = sorted(self.species)

        self.n_bits = [list(range(len(sl_bits))) for sl_bits in bits]
        if sl_sizes is None:
            self.sl_sizes = [1 for i in range(len(self.bits))]
        elif len(sl_sizes) == len(bits):
            self.sl_sizes = sl_sizes
        else:
            raise ValueError("Sublattice number mismatch: check bits \
                             and sl_sizes parameters.")

        self.N_sts_prim = sum(self.sl_sizes)

        (self.unit_n_excitations, self.chg_of_excitations,
         self.excitation_ids_by_sublat) = get_unit_excitations(self.bits)

        self._unit_basis = None
        self._unit_vertices = None

        # Matrices containing constrained subspace information.
        self._polytope = None

        # Minimum supercell size required to make vetices coordinates
        # all integer.
        self._min_sc_size = None
        self._min_int_vertices = None
        self._min_grid = None
        self._int_grids = {}

    @property
    def background_charge(self):
        """Charge of the background occupation state.

        Computed by summing charges of the last specie on each SiteSpace's.
        Return:
            Int.
        """
        chg = 0
        for sl_bits, sl_size in zip(self.bits, self.sl_sizes):
            chg += int(get_oxi_state(sl_bits[-1])) * sl_size
        return chg

    @property
    def unconstr_dim(self):
        """Dimensionality of the unconstrained space.

        Return:
            Int.
        """
        return len(self.unit_n_excitations)

    @property
    def is_charge_constred(self):
        """Check charge constrained.

        Whether or not this system has charge, and requires charge
        constraint. If true, charge neutrality reduces space dim by
        1.
        Return:
            Boolean.
        """
        d = len(self.chg_of_excitations)
        return not(np.allclose(np.zeros(d),
                   self.chg_of_excitations) and self.background_charge == 0)

    @property
    def dim(self):
        """Dimensionality of the constrained conpositional space.

        Return:
            Int.
        """
        d = self.unconstr_dim
        if not self.is_charge_constred:
            return d
        else:
            return d - 1

    @property
    def dim_nondisc(self):
        """Dimension of non-discriminative coordinates.

        Return:
            Int.
        """
        return len(self.species)

    @property
    def unit_basis(self):
        """Minimal charge-conserving flips in unconstrained coordinates.

        Get minimal charge neutral flips to span the whole compositional,
        space, in vector representation. These vectors are represented in
        unconstrained coordinates, and they are lattice basis vectors of
        the diophantine solutions.

        For example:
        [[Li+,Mn3+,Ti4+],[P3-,O2-]] system, minimal charge and number
        conserving flips are:
        3 Mn3+ <-> Li+ + 2 Ti4+, Ti4+ + P3- <-> Mn3+ + O2-
        Their vector forms are:
        (1,-3,0), (0,1,-1)

        Return:
            2D np.ndarray of np.int64
        """
        if self._unit_basis is None:
            self._unit_basis = np.array(get_integer_basis(
                                        self.chg_of_excitations,
                                        self.excitation_ids_by_sublat),
                                        dtype=np.int64)
        return self._unit_basis

    @property
    def min_flip_table(self):
        """Minimal charge conserving flips in table form.

        Return:
            List[Dict]
        """
        _flip_table = flip_vecs_to_flip_table(self.unit_n_excitations,
                                              self.n_bits,
                                              self.unit_basis)
        return _flip_table

    @property
    def min_flip_strings(self):
        """Reaction formulas of charge conserving flips.

        Intended to be human-readable.
        Return:
            List[str]
        """
        return visualize_flip_table(self.min_flip_table, self.bits)

    @property
    def polytope(self):
        """Constrained space in polytope form.

        Express the constrained configurational space as arrays necessary
        to initailize a polytope.Polytope object.

        Polytope is expressed by A @ x <= b.

        R and t are rotation matrix and translation vector to transform
        constrained to unconstrained basis.

        To transform a constrained basis to unconstrained basis, use:
            x = R.T @ x'.append(0) + t

        Notice: all A,b.R,t are defined in unitary (supercell size=1) space,
                therefore must normalize coordinates before applying these
                matrices.

        Return:
            tuple: (A, b, R, t), all np.ndarray.
        """
        if self._polytope is None:
            facets_unconstred = []
            for sl_flip_ids, sl_size in zip(self.excitation_ids_by_sublat,
                                            self.sl_sizes):
                if len(sl_flip_ids) == 0:
                    continue

                a = np.zeros(self.unconstr_dim)
                a[sl_flip_ids] = 1
                bi = sl_size
                facets_unconstred.append((a, bi))

            # sum(x_i) for i in sublattice <= 1
            A_n = np.vstack([a for a, bi in facets_unconstred])
            b_n = np.array([bi for a, bi in facets_unconstred])

            # x_i >=0 for all i
            A = np.vstack((A_n, -1 * np.identity(self.unconstr_dim)))
            b = np.concatenate((b_n, np.zeros(self.unconstr_dim)))

            if not self.is_charge_constred:
                # Polytope = pc.Polytope(A,b) Ax<=b.
                R = np.identity(self.unconstr_dim)
                t = np.zeros(self.unconstr_dim)
                self._polytope = (A, b, R, t)
            else:
                # x-t = R.T * x', where x'[-1]=0. Dimension reduced by 1.
                # We have to reduce dimension first because polytope package
                # can not handle polytope in a subspace. It will consider the
                # subspace as an empty set.

                # x: unconstrained, x': constrained
                R = np.vstack((self.unit_basis,
                              np.array(self.chg_of_excitations)))
                t = np.zeros(self.unconstr_dim)
                t[0] = -1 * self.background_charge / self.chg_of_excitations[0]
                A_sub = A @ R.T
                A_sub = A_sub[:, :-1]
                # Slice A, remove last col, because the last component of x'
                # will always be 0.
                b_sub = b - A @ t
                self._polytope = (A_sub, b_sub, R, t)

        return self._polytope

    @property
    def A(self):
        """Polytope inequality matrix.

        Return:
            np.ndarray
        """
        return self.polytope[0]

    @property
    def b(self):
        """Polytope inequality intercepts.

        Return:
            np.ndarray
        """
        return self.polytope[1]

    @property
    def R(self):
        """Rotation matrix for coordinates conversion.

        Return:
            np.ndarray
        """
        return self.polytope[2]

    @property
    def t(self):
        """T vector for coordinates conversion.

        Return:
            np.ndarray
        """
        return self.polytope[3]

    def _is_in_subspace(self, x, sc_size=1):
        """Check if unconstrained coords in charge neutral space.

        Args:
            x(1D arrayLike[int|float]):
                The unconstrained coordinate to examine.
            sc_size(int):
                Supercell size corresponding to the current coordinates.
        Return:
            Boolean.
        """
        x_scaled = np.array(x) / sc_size

        try:
            _ = self._unconstr_to_constr_coords(x_scaled, sc_size=1)
            return True
        except ValueError:
            return False

    def _constr_is_in_subspace(self, x, sc_size=1):
        """Check if unconstrained coords in charge neutral space.

        Args:
            x(1D arrayLike[int|float]):
                The constrained coordinates to check.
            sc_size(int, Default=1):
                Supercell size corresponding to the current coordinates.
        Return:
            Boolean.
        """
        return np.all(self.A @ (x / sc_size) <= self.b + SLACK_TOL)

    def unit_vertices(self, form='unconstr'):
        """Extremums of the constrained compositional space.

        All coordinates in a primitive cell.

        Args:
            form (string):
                Specifies the format to output the compositions.
                'unconstr': use unconstrained (type 1) coordinates.(default)
                            Return 2D np.ndarray.
                'constr': use constrained (type 2) coordinates.
                          Return 2D np.ndarray.
                'compstat': use compstat lists.(See self._unconstr_to_compstat)
                            Return List[List[List[int]]].
                'composition': use a pymatgen.composition for each sublattice
                               (vacancies not explicitly included)
                               Return List[List[Composition]].
        Return:
            Depends on form.
        """
        if self._unit_vertices is None:

            if not self.is_charge_constred:
                A, b, _, _ = self.polytope
                poly = pc.Polytope(A, b)
                self._unit_vertices = pc.extreme(poly)

            else:
                A, b, R, t = self.polytope
                poly_sub = pc.Polytope(A, b)
                vert_sub = pc.extreme(poly_sub)
                n = vert_sub.shape[0]
                vert = np.hstack((vert_sub, np.zeros((n, 1))))
                # Transform back into unconstraned coord
                self._unit_vertices = vert @ R + t

        if len(self._unit_vertices) == 0:
            raise CHGBALANCEERROR

        # This function formuates multiple unconstrained coords together.
        return self._convert_unconstr_to(self._unit_vertices, form=form,
                                         sc_size=1)

    def get_random_point_in_unit_space(self, form='unconstr'):
        """Random point inside the prim, charge-neutral space.

        Args:
            form(str):
                Desired format of output.
                'unconstr': use unconstrained (type 1) coordinates.(default)
                            Return np.ndarray.
                'constr': use constrained (type 2) coordinates.
                          Return np.ndarray.
                'compstat': use compstat lists.(See self._unconstr_to_compstat)
                            Return List[List[float]].
                'composition': use a pymatgen.composition for each sublattice
                               (vacancies not explicitly included)
                               Return List[Composition]].
        Return:
            Depends on form.
        """
        verts = self.unit_vertices(form='unconstr')
        x = verts[0].copy()

        for i in range(1, len(verts)):
            lam = np.random.random()
            x = lam*x + (1 - lam) * verts[i]

            in_spc = True
            if not self.is_charge_constred:
                x_prime = deepcopy(x)
            else:
                x_prime = np.linalg.inv((self.R).T) @ (x - self.t)
                # d_slack = x_prime[-1]
                x_prime = x_prime[:-1]

            b = self.A @ x_prime
            for bi_p, bi in zip(b, self.b):
                if bi_p - bi > -1 * SLACK_TOL:
                    in_spc = False
                    break

            if in_spc:
                break

        return self._convert_unconstr_to(x, form=form, sc_size=1)

    @property
    def min_sc_size(self):
        """Minimal supercell size to have a integer composition.

        Return:
            Int.
        """
        if self._min_sc_size or self._min_int_vertices is None:
            self._min_int_vertices, self._min_sc_size \
                = integerize_multiple(self.unit_vertices())
        return self._min_sc_size

    def min_int_vertices(self, form='unconstr'):
        """Minimal integerized compositional space vertices.

        Args:
            form (string):
                Desired format of output.
                'unconstr': use unconstrained (type 1) coordinates.(default)
                            Return 2D np.ndarray.
                'constr': use constrained (type 2) coordinates.
                          Return 2D np.ndarray.
                'compstat': use compstat lists.(See self._unconstr_to_compstat)
                            Return List[List[List[int]]].
                'composition': use a pymatgen.composition for each sublattice
                               (vacancies not explicitly included)
                               Return List[List[Composition]]].
        Return:
            Depends on form.
        """
        if self._min_sc_size or self._min_int_vertices is None:
            _ = self.min_sc_size

        return self._convert_unconstr_to(self._min_int_vertices,
                                         form=form, sc_size=self.min_sc_size)

    def int_vertices(self, sc_size=1, form='unconstr'):
        """Integer vertices in a neutral space of specific supercell size.

        If supercell size is a multiple of min_sc_size, then int_vertices are
        just min_int_vertices*multiple. Otherwise int_vertices are taken as
        convex hull vertices of set self.int_grids(sc_size)

        Args:
            sc_size(int):
                supercell sizes to enumerate integer composition on
            form (string):
                Desired format of output.
                'unconstr': use unconstrained (type 1) coordinates.(default)
                            Return 2D np.ndarray.
                'constr': use constrained (type 2) coordinates.
                          Return 2D np.ndarray.
                'compstat': use compstat lists.(See self._unconstr_to_compstat)
                            Return List[List[List[int]]].
                'composition': use a pymatgen.composition for each sublattice
                               (vacancies not explicitly included)
                               Return List[List[Composition]]].
        Return:
            Depends on form.
        """
        if sc_size % self.min_sc_size == 0:
            vertices = self.min_int_vertices() * (sc_size // self.min_sc_size)
        else:
            # Approximate bounding vertices.
            bound_vertices = (self.min_int_vertices(form='constr')
                              * (sc_size / self.min_sc_size))

            # Since constrained coords are no longer ints, we have to shift
            # by a basis solution.
            base_shift = np.array(get_integer_base_solution(
                                  self.chg_of_excitations,
                                  right_side=-1 * self.background_charge
                                  * sc_size))
            base_shift = (np.linalg.inv((self.R).T) @
                          (base_shift / sc_size - self.t) * sc_size)[:-1]

            vertices_estimate = []
            # Shift, jitter, then shift back.
            for v in (bound_vertices - base_shift):
                v_floor = np.floor(v)
                shifts = np.array(list(product(*[[0, 1]
                                  for i in range(len(v))])))
                v_estimate = shifts + v_floor + base_shift
                vertices_estimate.append(v_estimate)
            vertices_estimate = np.vstack(vertices_estimate)

            estimates_in_space = [self._constr_is_in_subspace(v,
                                                              sc_size=sc_size)
                                  for v in vertices_estimate]
            vertices_estimate = vertices_estimate[estimates_in_space]

            try:
                hull = ConvexHull(vertices_estimate)
                vertices = vertices_estimate[hull.vertices]
            except QhullError:
                vertices = vertices_estimate
            vertices = self._convert_to_unconstr(vertices, form='constr',
                                                 sc_size=sc_size)

        vertices = np.array(np.round(vertices), dtype=np.int64)

        return self._convert_unconstr_to(vertices, form=form, sc_size=sc_size)

    def min_grid(self, form='unconstr'):
        """All charge-neutral integer compositions in under min_sc_size.

        Args:
            form (string):
                Desired format of output.
                'unconstr': use unconstrained (type 1) coordinates.(default)
                            Return 2D np.ndarray.
                'constr': use constrained (type 2) coordinates.
                          Return 2D np.ndarray.
                'compstat': use compstat lists.(See self._unconstr_to_compstat)
                            Return List[List[List[int]]].
                'composition': use a pymatgen.composition for each sublattice
                               (vacancies not explicitly included)
                               Return List[List[Composition]]].
        Return:
            Depends on form.
        """
        if self._min_grid is None:
            self._min_grid = self._enum_int_grids(sc_size=self.min_sc_size)

        return self._convert_unconstr_to(self._min_grid,
                                         form=form, sc_size=self.min_sc_size)

    def int_grids(self, sc_size=1, form='unconstr'):
        """All integer compositions in a supercell with specified size.

        Args:
            sc_size(int):
                supercell sizes to enumerate integer composition on
            form (string):
                Desired format of output.
                'unconstr': use unconstrained (type 1) coordinates.(default)
                            Return 2D np.ndarray.
                'constr': use constrained (type 2) coordinates.
                          Return 2D np.ndarray.
                'compstat': use compstat lists.(See self._unconstr_to_compstat)
                            Return List[List[List[int]]].
                'composition': use a pymatgen.composition for each sublattice
                               (vacancies not explicitly included)
                               Return List[List[Composition]]].
        Return:
            Depends on form.

        Note: if you want enumeration by a step!=1, just divide sc_size by that
              step, and multiply the resulted array with step. (You don't need
              to multiply back, when formula = 'composition' since it's always
              normalized.)
        """
        if sc_size not in self._int_grids:
            self._int_grids[sc_size] = self._enum_int_grids(sc_size=sc_size)

        return self._convert_unconstr_to(self._int_grids[sc_size],
                                         form=form, sc_size=sc_size)

    def _enum_int_grids(self, sc_size=1):
        """Enumerate all possible integer compositions.

        Gives unconstrained coordinates only.

        Args:
            sc_size(int):
                The supercell size to enumerate integer compositions on.
                Recommended is a multiply of self.min_sc_size, otherwise
                we can't guarantee to find any integer composition.
        Return:
            np.ndarray
        """
        if sc_size % self.min_sc_size == 0:
            magnif = sc_size // self.min_sc_size
            int_vertices = self.min_int_vertices() * magnif
            limiters_ub = np.max(int_vertices, axis=0)
            limiters_lb = np.min(int_vertices, axis=0)

        else:
            # Then integer composition is not guaranteed to be found.
            vertices = self.unit_vertices() * sc_size
            limiters_ub = np.array(np.ceil(np.max(vertices, axis=0)),
                                   dtype=np.int64)
            limiters_lb = np.array(np.floor(np.min(vertices, axis=0)),
                                   dtype=np.int64)

        limiters = list(zip(limiters_lb, limiters_ub))
        right_side = -1 * self.background_charge * sc_size
        grid = get_integer_grid(self.chg_of_excitations, right_side=right_side,
                                limiters=limiters)

        enum_grid = []
        for p in grid:
            if self._is_in_subspace(p, sc_size=sc_size):
                enum_grid.append(p)

        return np.array(enum_grid, dtype=np.int64)

    def frac_grids(self, sc_size=1, form='unconstr'):
        """Normalize neutral compositions with a supercell size.

        Enumerates integer compositions under a certain sc_size, and normalize
        with sc_size. ('composition' format normalized with sublattice sizes.)

        Args:
            sc_size(int):
                supercell sizes to enumerate integer composition on
            form (string):
                Desired format of output.
                'unconstr': use unconstrained (type 1) coordinates.(default)
                            Return 2D np.ndarray.
                'constr': use constrained (type 2) coordinates.
                          Return 2D np.ndarray.
                'compstat': use compstat lists.(See self._unconstr_to_compstat)
                            Return List[List[List[int]]].
                'composition': use a pymatgen.composition for each sublattice
                               (vacancies not explicitly included)
                               Return List[List[Composition]]].

        Return:
            Depends on form.

        Note: if you want to stepped enumeration, just divide sc_size by step,
              and enter into sc_size argument.
        """
        comps = np.array(self.int_grids(sc_size), dtype=np.float64) / sc_size

        return self._convert_unconstr_to(comps, form=form, sc_size=1)

# These formatting functions will not normalize or scale compositions.
# It's your responsibility to check scales are correct.
    def _unconstr_to_constr_coords(self, x, sc_size=1, to_int=False):
        """Unconstrained coordinates to constrained coordinates.

        Args:
            x(1D Arraylike):
                Unconstrained coordinates.
            sc_size(int):
                Supercell size corresponding to the given coordinates.
            to_int(Boolean):
                If true, round coords to integers.

        Return:
            np.ndarray
        """
        # Scale down to unit comp space
        x = np.array(x) / sc_size

        if not self.is_charge_constred:
            x_prime = deepcopy(x)
            d_slack = 0
        else:
            x_prime = np.linalg.inv((self.R).T) @ (x - self.t)
            d_slack = x_prime[-1]
            x_prime = x_prime[:-1]

        b = self.A @ x_prime

        # Check if the given coordinates are in constrained subspace.
        for bi_p, bi in zip(b, self.b):
            if bi_p - bi > SLACK_TOL:
                raise OUTOFSUBSPACEERROR

        if abs(d_slack) > SLACK_TOL:
            raise OUTOFSUBSPACEERROR

        # Scale back up to sc_size
        x_prime = x_prime * sc_size
        d_slack = d_slack * sc_size

        if to_int:
            x_prime = np.round(x_prime)
            x_prime = np.array(x_prime, dtype=np.int64)

        return x_prime

    def _constr_to_unconstr_coords(self, x_prime, sc_size=1, to_int=False):
        """Constrained coordinates to unconstrained coordinates.

        Args:
            x_prime(1D Arraylike):
                Constrained coordinates.
            sc_size(int):
                Supercell size corresponding to the given coordinates.
            to_int(Boolean):
                If true, round coords to integers.

        Return:
            np.ndarray
        """
        # Scale down to unit comp space.
        x_prime = np.array(x_prime) / sc_size

        b = self.A @ x_prime
        for bi_p, bi in zip(b, self.b):
            if bi_p - bi > SLACK_TOL:
                raise OUTOFSUBSPACEERROR

        if not self.is_charge_constred:
            x = deepcopy(x_prime)
        else:
            x = deepcopy(x_prime)
            x = np.concatenate((x, np.array([0])))
            x = (self.R).T @ x + self.t

        # Scale back up
        x = x * sc_size

        if to_int:
            x = np.round(x)
            x = np.array(x, dtype=np.int64)

        return x

    def _unconstr_to_compstat(self, x, sc_size=1):
        """Unconstrained coordinates to compstats.

        Args:
            x(1D arrayLike):
                Unconstrained coordinates.
            sc_size(int):
                Supercell size corresponding to given coordinates.

        Return:
            List[List[int]]
        """
        v_id = 0
        compstat = [[0 for i in range(len(sl_n_bits))]
                    for sl_n_bits in self.n_bits]

        for sl_id, sl_n_bits in enumerate(self.n_bits):
            sl_sum = 0
            for b_id, bit in enumerate(sl_n_bits[:-1]):
                compstat[sl_id][b_id] = x[v_id]
                sl_sum += x[v_id]
                v_id += 1

            compstat[sl_id][-1] = self.sl_sizes[sl_id] * sc_size - sl_sum

            if compstat[sl_id][-1] < 0:
                raise OUTOFSUBSPACEERROR

        return compstat

    def _compstat_to_unconstr(self, compstat):
        """Compstat to unconstrained coordinates.

        Args:
            compstat(List[List[int]]):
                Species counts on each sublattices.
        Return:
            1D np.ndarray
        """
        x = []
        for sl_stat in compstat:
            x.extend(sl_stat[:-1])
        return np.array(x)

    def _unconstr_to_composition(self, x, sc_size=1):
        """Unconstrained coordinates to composition format.

        Translate an unconstranied coordinate into a list of Composition
        by each sublattice. Vacancies are not explicitly included for the
        convenience structure generation.

        Args:
            x(1D arrayLike):
                Unconstrained coordinates.
            sc_size(int):
                Supercell size corresponding to the coordinates.

        Return:
            List[Composition]
        """
        compstat = self._unconstr_to_compstat(x, sc_size=sc_size)

        sl_comps = []
        for sl_id, sl_bits in enumerate(self.bits):
            sl_comp = {}

            for b_id, bit in enumerate(sl_bits):
                # Trim vacancies from the composition, for pymatgen to
                # read the structure.
                if isinstance(bit, Vacancy):
                    continue
                # Composition will always be normalized!
                sl_comp[bit] = (compstat[sl_id][b_id] /
                                (self.sl_sizes[sl_id] * sc_size))

            sl_comps.append(Composition(sl_comp))

        return sl_comps

    def _composition_to_unconstr(self, comp, sc_size=1):
        """Composition format to unconstrained coordinates.

        Args:
            comp(List[Composition]) :
                Composition format.
            sc_size(int):
                Supercell size corresponding to this compositon.

        Return:
            np.ndarray
        """
        x = []

        for sl_id, sl_bits in enumerate(self.bits):
            sl_size = self.sl_sizes[sl_id] * sc_size
            for b_id, bit in enumerate(sl_bits[:-1]):
                if isinstance(bit, Vacancy):
                    sl_sum = sum(list(comp[sl_id].values())) * sl_size
                    if sl_sum > sl_size:
                        raise ValueError("{} is not a valid composition."
                                         .format(comp[sl_id]))

                    x.append(sl_size - sl_sum)
                else:
                    x.append(comp[sl_id][bit] * sl_size)

        return np.array(x)

    def _unconstr_to_nondisc(self, x, sc_size=1):
        """Unconstrained coordinates to non-discriminative (irreversible).

        Translates an unconstrained coordinate into non-discriminative
        coordinate. Same specie on different sublattices will be summed
        up as one coordinate.

        Args:
            x(1D arrayLike):
                Unconstrained cooordinate.
            sc_size(int):
                Supercell size.

        Return:
            np.ndarray
        """
        nondisc = np.zeros(self.dim_nondisc)
        compstat = self._unconstr_to_compstat(x, sc_size=sc_size)

        for sl_bits, sl_ns in zip(self.bits, compstat):
            for b, n in zip(sl_bits, sl_ns):
                for sp_id, sp in enumerate(self.species):
                    if sp == b:
                        nondisc[sp_id] += n
                        break
                    elif (isinstance(sp, Vacancy) and isinstance(b, Vacancy)):
                        nondisc[sp_id] += n
                        break

        if np.sum(nondisc) != sc_size * sum(self.sl_sizes):
            raise OUTOFSUBSPACEERROR

        return nondisc

    def _convert_unconstr_to(self, x, form='unconstr', sc_size=1):
        """Unconstrained coordinates to any specified form.

        Args:
            x(1D ArrayLike):
                Unconstrained coordinates.
            sc_size (int):
                Supercell size to numerate on.
            form (string):
                Specifies the format to output the compositions.
                'unconstr': use unconstrained (type 1) coordinates.(default)
                'constr': use constrained (type 2) coordinates.
                'compstat': use compstat lists.(See self._unconstr_to_compstat)
                'composition': use a pymatgen.composition for each sublattice
                               (vacancies not explicitly included)
                'nondisc': non-discriminate coordinates.
        Return:
            Depends on form.
        """
        if len(np.array(x).shape) > 1:
            result = [self._convert_unconstr_to(x_sub, form=form,
                      sc_size=sc_size) for x_sub in x]
            if form in ['constr', 'unconstr', 'nondisc']:
                return np.vstack(result)
            else:
                return result

        if form == 'unconstr':
            return np.array(x)
        elif form == 'constr':
            return np.array(self._unconstr_to_constr_coords(x,
                            sc_size=sc_size))
        elif form == 'compstat':
            return self._unconstr_to_compstat(x, sc_size=sc_size)
        elif form == 'composition':
            return self._unconstr_to_composition(x, sc_size=sc_size)
        elif form == 'nondisc':
            return self._unconstr_to_nondisc(x, sc_size=sc_size)
        else:
            raise ValueError('Requested format not supported.')

    def _convert_to_unconstr(self, x, form, sc_size=1):
        """Different forms into unconstrained coordinates.

        Args:
            x:
                Input composition coordinates, format depends on form.
            sc_size (int):
                Supercell size to numerate on.
            form (string):
                Specifies the format pf input.
                'unconstr': use unconstrained (type 1) coordinates.(default)
                'constr': use constrained (type 2) coordinates.
                'compstat': use compstat lists.(See self._unconstr_to_compstat)
                'composition': use a pymatgen.composition for each sublattice
                               (vacancies not explicitly included, and must be
                                normalized.)
        Return:
            np.ndarray

        Note: 'non-disc' can not be converted back to 'unconstr'.
        """
        if form in ['unconstr', 'constr']:
            if len(np.array(x).shape) > 1:
                result = [self._convert_to_unconstr(x_sub,
                          form=form, sc_size=sc_size)
                          for x_sub in x]
                return np.vstack(result)

        elif form == 'compstat':
            if isinstance(x[0][0], list):
                result = [self._convert_to_unconstr(x_sub,
                          form=form, sc_size=sc_size)
                          for x_sub in x]
                return np.vstack(result)

        elif form == 'composition':
            if not isinstance(x[0], Composition):
                result = [self._convert_to_unconstr(x_sub,
                          form=form, sc_size=sc_size)
                          for x_sub in x]
                return np.vstack(result)

        if form == 'unconstr':
            return np.array(x)
        elif form == 'constr':
            return np.array(self._constr_to_unconstr_coords(x,
                            sc_size=sc_size))
        elif form == 'compstat':
            return np.array(self._compstat_to_unconstr(x))
        elif form == 'composition':
            # Output will be a 2D list of pymatgen.Composition
            return np.array(self._composition_to_unconstr(x,
                            sc_size=sc_size))
        else:
            raise ValueError('Requested format not supported.')

    def translate_format(self, x, from_format, to_format='unconstr',
                         sc_size=1):
        """Translate between compositional coordinates forms.

        Args:
            x:
                Input compositional coordinates. Format depends on
                'from_format' argument.
            sc_size (int):
                Supercell size to numerate on.
                Default is 1, we suppose you are using primitive cell scale.
            from_format,to_format (string):
                Specifies the format of input and output compositions.
                'unconstr': use unconstrained (type 1) coordinates.(default)
                'constr': use constrained (type 2) coordinates.
                'compstat': use compstat lists.(See self._unconstr_to_compstat)
                'composition': use a pymatgen.composition for each sublattice
                               (vacancies not explicitly included)
                'nondisc': Non discriminative compositional coordinates.
                           Can only be taken by 'to_format' argument.
        Return:
            Depends on 'to_format' argument.
        """
        if from_format == 'nondisc':
            raise ValueError('Non-discriminative coordinates can not be \
                             converted to discriminative!')

        ucoords = self._convert_to_unconstr(x, form=from_format,
                                            sc_size=sc_size)
        return self._convert_unconstr_to(ucoords, form=to_format,
                                         sc_size=sc_size)

    def as_dict(self):
        """Serialize into dictionary.

        Return:
            Dict.
        """
        bits_d = [[sp.as_dict() for sp in sl_sps] for sl_sps in self.bits]

        poly = [item.tolist() for item in self.polytope]
        int_grids = {key: val.tolist() for key, val
                     in self._int_grids.items()}

        return {
                'bits': bits_d,
                'sl_sizes': self.sl_sizes,
                'unit_basis': self.unit_basis.tolist(),
                'unit_vertices': self.unit_vertices().tolist(),
                'polytope': poly,
                'min_sc_size': self.min_sc_size,
                'min_int_vertices': self.min_int_vertices().tolist(),
                'min_grid': self.min_grid().tolist(),
                'int_grids': int_grids,
                '@module': self.__class__.__module__,
                '@class': self.__class__.__name__
               }

    @classmethod
    def from_dict(cls, d):
        """Load CompSpace object from dictionary.

        Args:
            d(dict):
                Dictionary to decode from.
        Return:
            CompSpace
        """
        bits = [[MontyDecoder().process_decoded(sp_d) for sp_d in sl_sps]
                for sl_sps in d['bits']]

        obj = cls(bits, d['sl_sizes'])

        if 'unit_basis' in d:
            obj._unit_basis = np.array(d['unit_basis'],
                                       dtype=np.int64)

        if 'unit_vertices' in d:
            obj._unit_vertices = np.array(d['unit_vertices'])

        if 'polytope' in d:
            poly = d['polytope']
            poly = [np.array(item) for item in poly]
            obj._polytope = poly

        if 'min_sc_size' in d:
            obj._min_sc_size = d['min_sc_size']

        if 'min_int_vertices' in d:
            obj._min_int_vertices = np.array(d['min_int_vertices'],
                                             dtype=np.int64)

        if 'min_grid' in d:
            obj._min_grid = np.array(d['min_grid'], dtype=np.int64)

        if 'int_grids' in d:
            int_grids = d['int_grids']
            obj._int_grids = {key: np.array(val, dtype=np.int64)
                              for key, val in int_grids}

        return obj
