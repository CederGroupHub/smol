"""Defines a space of compositions."""

__author__ = "Fengyu Xie"

import warnings
import numpy as np
from itertools import chain
from monty.json import MSONable, MontyDecoder

from pymatgen.core import Element, Composition
from smol.cofe.space.domain import Vacancy

from .utils.math_utils import (NUM_TOL, get_nonneg_float_vertices,
                               integerize_multiple, solve_diophantines,
                               get_natural_solutions, get_optimal_basis,
                               get_ergodic_vectors, get_natural_centroid)
from .utils.occu_utils import get_dim_ids_by_sublattice


class NegativeSpeciesError(Exception):
    """Raises when count of species is negative."""

    def __init__(self, c, form,
                 message="Composition results in "
                         "negative species count!"):
        """Initialize exception class."""
        self.c = c
        self.form = form
        self.message = message
        super(NegativeSpeciesError, self).__init__(self.message)

    def __str__(self):
        """Represent."""
        return f"Composition: {self.c}, format: {self.form} " \
               f"-> {self.message}"


class RoundingError(Exception):
    """Raises when count of species can not be rounded to integers."""

    def __init__(self, c, form,
                 message="Composition can not be rounded "
                         "to integer!"):
        """Initialize exception class."""
        self.c = c
        self.form = form
        self.message = message
        super(RoundingError, self).__init__(self.message)

    def __str__(self):
        """Represent."""
        return f"Composition: {self.c}, format: {self.form}, " \
               f"tolerance: {NUM_TOL} -> {self.message}"


class ConstraintViolationError(Exception):
    """Raises when count of species violate constraints."""

    def __init__(self, c, form,
                 message="Composition violates constraints!"):
        """Initialize exception class."""
        self.c = c
        self.form = form
        self.message = message
        super(ConstraintViolationError, self).__init__(self.message)

    def __str__(self):
        """Represent."""
        return f"Composition: {self.c}, format: {self.form}, " \
               f"tolerance: {NUM_TOL} -> {self.message}"


class CompUnNormalizedError(Exception):
    """Raises when pymatgen Composition is not normalized to 1."""

    def __init__(self, c,
                 message="Composition in comp format but not normalized!"):
        """Initialize exception class."""
        self.c = c
        self.message = message
        super(CompUnNormalizedError, self).__init__(self.message)

    def __str__(self):
        """Represent."""
        return f"Composition: {self.c} -> {self.message}"


def get_oxi_state(sp):
    """Oxidation state from Specie/Element/Vacancy.

    Args:
       sp(Specie/Vacancy/Element):
          A species.
    Return:
       Charge of species: int.
    """
    if isinstance(sp, Element):
        return 0
    else:
        return int(sp.oxi_state)


def flip_vec_to_reaction(u, bits):
    """Convert flip direction into a reaction formula in string.

    This function is for easy interpretation of flip directions.
    Args:
        u(1D ArrayLike[int]):
            The flip vector in number change of species.
        bits(List[List[Specie|DummySpecie|Element|Vacancy]]):
            Species on all sub-lattices.
    Return:
        Reaction formulas: str.
    """
    u = np.array(u, dtype=int)
    dim_ids = get_dim_ids_by_sublattice(bits)

    from_strs = []
    to_strs = []

    for sl_id, (sl_species, sl_dims) in enumerate(zip(bits, dim_ids)):
        for specie, dim in zip(sl_species, sl_dims):
            if u[dim] < 0:
                from_strs.append('{} {}({})'
                                 .format(-u[dim], str(specie), sl_id))
            elif u[dim] > 0:
                to_strs.append('{} {}({})'
                               .format(u[dim], str(specie), sl_id))

    from_str = ' + '.join(from_strs)
    to_str = ' + '.join(to_strs)
    return from_str + ' -> ' + to_str


class CompSpace(MSONable):
    """Composition space class.

    Generates a charge neutral compositional space from a list of Species
    or DummySpecies and number of sites in each sub-lattice of a PRIM CELL.

    A composition can be expressed in 4 formats:
    1, Number of species on each sub-lattice, concatenated by the
       order of each sub-lattice into an 1D array. ("n" format)

    2, Coordinates x on the constrained integer grid. n = n0 + V @ x,
       where V are grid basis and n0 is a base solution (whose n is
       not necessarily guaranteed as natural numbers).
       ("x" format).

    3, pymatgen.Composition of each sub-lattice, normalized by number of
       sites in the sub-lattice. Vacancies never explicitly included.
       ("comp" format).

       Note: This format is always normalized (0<num_atoms <=1).
       If <1, the remaining part will be filled with vacancies.

    4, Count of sorted species without distinguishing sub-lattices.
       Namely, if one species appears in multiple sub-lattices, it will
       only be counted into one component of the vector correponding to
       that species. ("nondisc" format).

    Attributes:
        n_dims(int):
            Sum of species number in all sub-lattices. Namely,
            the dimension in "n" format.
        species(List[Species|DummySpecies|Element|Vacancy]):
            All species in the given system (sorted).
        dim_ids(List[List[int]]):
            The corresponding index in the "n" vector
            of each species on each sub-lattice.
        dim_ids_nondisc(List[List[int]]):
            The corresponding index in the "nondisc" vector
            of each species on each sub-lattice.
    """

    def __init__(self, bits, sl_sizes=None,
                 charge_balanced=True,
                 other_constraints=None,
                 optimize_basis=False,
                 table_ergodic=False):
        """Initialize CompSpace.

        Args:
            bits(List[List[Specie|Vacancy|Element]]):
                Species on each sub-lattice.
            sl_sizes(1D ArrayLike[int]): optional
                Number of sites in each sub-lattice per primitive cell.
                If not given, assume one site for each sub-lattice.
                Better provide them as co-prime integers.
            charge_balanced(bool): optional
                Whether to add charge balance constraint. Default
                to true.
            other_constraints(List[tuple(1D arrayLike[int], int)]): optional
                Other integer constraints except charge balance and
                site-number conservation. Should be given in the form of
                tuple(a, bb), each gives constraint np.dot(a, n)=bb.
            optimize_basis(bool): optional
                Whether to optimize the basis to minimal flip sizes and maximal
                connectivity in the minimum super-cell size.
                When the minimal super-cell size is large, we recommend not to
                optimize basis.
            table_ergodic(bool): optional
                When generating a flip table, whether to add vectors and
                ensure ergodicity under a minimal super-cell size.
                Default to False.
                When the minimal super-cell size is large, we recommend not to
                ensure ergodicity. This is not only because of the computation
                difficulty; but also because at large super-cell size,
                the fraction of inaccessible compositions usually becomes
                minimal.
        """
        self.bits = bits
        self.n_dims = sum([len(species) for species in bits])
        self.dim_ids = get_dim_ids_by_sublattice(self.bits)
        # dimension of "n" format

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
        self.species = sorted(self.species)

        dim_ids_nondisc = []
        for species in self.bits:
            sl_dim_ids = []
            for sp in species:
                if not isinstance(sp, Vacancy):
                    sl_dim_ids.append(self.species.index(sp))
                else:
                    for sp2_id, sp2 in enumerate(self.species):
                        if isinstance(sp2, Vacancy):
                            sl_dim_ids.append(sp2_id)
                            break
            dim_ids_nondisc.append(sl_dim_ids)
        self.dim_ids_nondisc = dim_ids_nondisc

        if sl_sizes is None:
            self.sl_sizes = [1 for _ in range(len(self.bits))]
        elif len(sl_sizes) == len(bits):
            self.sl_sizes = np.array(sl_sizes, dtype=int).tolist()
        else:
            raise ValueError("Sub-lattice number is not the same "
                             "in parameters bits and sl_sizes.")

        self.charge_balanced = charge_balanced
        self.optimize_basis = optimize_basis
        self.table_ergodic = table_ergodic

        # Set constraint equations An=b (per primitive cell).
        A = []
        b = []
        if charge_balanced:
            A.append([get_oxi_state(sp)
                      for species in bits for sp in species])
            b.append(0)
        for dim_id, sl_size in zip(self.dim_ids, self.sl_sizes):
            a = np.zeros(self.n_dims, dtype=int)
            a[dim_id] = 1
            A.append(a.tolist())
            b.append(sl_size)
        if other_constraints is None:
            other_constraints = []
        for a, bb in other_constraints:
            if len(a) != self.n_dims:
                raise ValueError(f"Constraint length: {len(a)} does not match"
                                 f" dimensions: {self.n_dims}!")
            if not np.allclose(np.array(a), np.round(a)) or not \
                    np.allclose(b, np.round(b)):
                raise ValueError("Provided constraints must be integers!")
            A.append(a)
            b.append(bb)
        self._A = np.array(np.round(A), dtype=int)
        self._b = np.array(np.round(b), dtype=int)

        self._prim_vertices = None
        # Minimum supercell size required to make prim_vertices coordinates
        # all integer.
        self._min_sc_size = None
        self._flip_table = None
        self._n0 = None  # Prim cell base solution
        self._vs = None  # Basis vectors, n = n0 + vs.T @ x
        self._comp_grids = {}  # Grid of integer compositions, in "x" format.

    @property
    def A(self):
        """Matrix A in constraints An=b.

        Returns:
            2D np.ndarray[int]
        """
        return self._A

    @property
    def b(self):
        """Vector b in constraints An=b.

        Returns:
            1D np.ndarray[int]
        """
        return self._b

    @property
    def prim_vertices(self):
        """Vertices of polytope An=b, n>=0 in prim cell.

        Returns:
            prim vertices in "n" format:
                2D np.ndarray[float]
        """
        if self._prim_vertices is None:
            self._prim_vertices = get_nonneg_float_vertices(self.A, self.b)
        return self._prim_vertices

    @property
    def min_sc_size(self):
        """Minimum integer super-cell size.

        Returns:
            int
        """
        if self._min_sc_size is None:
            int_verts, sc_size = integerize_multiple(self.prim_vertices)
            self._min_sc_size = sc_size
        return self._min_sc_size

    @property
    def n_comps_estimate(self):
        """Estimated number of compositions w.o constraints."""
        return np.prod([(sl_size * self.min_sc_size) ** len(species)
                        for species, sl_size
                        in zip(self.bits, self.sl_sizes)])

    @property
    def n0(self):
        """Primitive cell base solution (not natural numbers).

        self.n0 * sc_size is a base solution in super-cell in
        "n" format.
        Returns:
             1D np.ndarray[int]
        """
        if self._n0 is None:
            n0, vs = solve_diophantines(self.A, self.b)
            self._n0 = n0
        return self._n0

    @property
    def basis(self):
        """Basis vectors of the composition grid An=b.

        Will be optimized if self.optimize_basis is set
        to true.
        Returns:
            Basis vectors in "n" format:
                2D np.ndarray[int]
        """
        if self._vs is None:
            n0, vs = solve_diophantines(self.A,
                                        self.b * self.min_sc_size)
            if self.optimize_basis:
                n_comps = self.n_comps_estimate
                if n_comps > 10 ** 6:
                    warnings.warn("Basis optimization can be very costly "
                                  "at your composition space size = "
                                  f"{n_comps}. "
                                  "Do this at your own risk!")
                xs = get_natural_solutions(n0, vs)
                vs_opt = get_optimal_basis(n0, vs, xs)
            else:
                vs_opt = vs.copy()
            self._vs = vs_opt
        return self._vs

    @property
    def flip_table(self):
        """Give flip table vectors.

        If self.table_ergodic is true, will add flips if basis
        is not ergodic.
        Returns:
            Flip vectors in the "n" format:
                2D np.ndarray[int]
        """
        if self._flip_table is None:
            if not self.table_ergodic:
                self._flip_table = self.basis.copy()
            else:
                n_comps = self.n_comps_estimate
                if n_comps > 10 ** 6:
                    warnings.warn("Ergodicity computation can be very costly "
                                  "at your composition space size = "
                                  f"{n_comps}. "
                                  "Do this at your own risk!")
                self._flip_table = get_ergodic_vectors(self.n0
                                                       * self.min_sc_size,
                                                       self.basis,
                                                       self.min_sc_grid)
        return self._flip_table

    @property
    def flip_reactions(self):
        """Reaction formulae of table flips.

        Return:
            Reaction formulae (only forward direction):
                List[str]
        """
        return [flip_vec_to_reaction(u, self.bits) for u in self.flip_table]

    def get_comp_grid(self, sc_size=1):
        """Get the natural number compositions.

        Args:
            sc_size(int):
                Super-cell size to enumerate with.
        Returns:
            Solutions to n0 + Vx >= 0 ("x" format, not normalized):
                2Dnp.ndarray[int]
        """
        if sc_size not in self._comp_grids:
            self._comp_grids[sc_size] = get_natural_solutions(self.n0
                                                              * sc_size,
                                                              self.basis)
        return self._comp_grids[sc_size]

    @property
    def min_sc_grid(self):
        """Get the natural number solutions on grid at min_sc_size.

        Returns:
            Solutions to n0 + Vx >= 0 ("x" format, not normalized):
                2Dnp.ndarray[int]
        """
        return self.get_comp_grid(sc_size=self.min_sc_size)

    def get_centroid_composition(self, sc_size=None):
        """Get a composition close to the centroid of polytope.

        Args:
            sc_size(int): optional
               Super-cell size to get the composition with.
               If not given, will use self.min_sc_size
        Return:
            Composition close to centroid of polytope n0 + Vx >= 0
            ("x" format, not normalized):
                1D np.ndarray[int]
        """
        if sc_size is None:
            sc_size = self.min_sc_size
        return get_natural_centroid(self.n0 * sc_size, self.basis)

    def translate_format(self, c, sc_size, from_format, to_format='n',
                         rounding=False):
        """Translate between composition formats.

        Args:
            c(1D ArrayLike|List[Composition]):
                Input format. Can be int or fractional.
            sc_size (int):
                Supercell size of this composition. You must make sure
                it is the right super-cell size of composition c.
            from_format(str):
                Specifies the input format.
                "n": number of species on each sub-lattice, concatenated;
                "x": the grid coordinates;
                "comp": pymatgen.Composition on each sub-lattice, Vacancies
                        not explicitely included.
                "nondisc" can not be the input format.
                "n" and "x" requires un-normalized, while "comp" is normalized
                to number of sites in sub-lattice.
            to_format(str): optional
                Specified the output format.
                Same as from_format, with an addition:
                "nondisc": count of each species, regardless of sub-lattice.
                If not give, will always convert to "n" format.
            rounding(bool): optional
                If the returned format is "n", "x" or "nondisc", whether
                or not to round up the output array as integers. Default to
                false.
        Return:
            Depends on to_format argument ("n","x","nondisc" not normalized):
                1D np.ndarray[int|float]|List[Composition]
        """
        if from_format == 'nondisc':
            raise ValueError("nondisc format can not be converted to "
                             "other formats!")

        n = self._convert_to_n(c, form=from_format, sc_size=sc_size,
                               rounding=rounding)
        return self._convert_n_to(n, form=to_format, sc_size=sc_size,
                                  rounding=rounding)

    def _convert_to_n(self, c, form, sc_size, rounding):
        """Convert other composition format to n-format."""
        if form == "n":
            n = np.array(c)
        elif form == "x":
            n = self.basis.transpose() @ np.array(c) + self.n0 * sc_size
        elif form == "comp":
            n = []
            for species, sl_size, comp in zip(self.bits, self.sl_sizes, c):
                if comp.num_atoms > 1 or comp.num_atoms < 0:
                    raise CompUnNormalizedError(comp)
                vac_counted = False
                for specie in species:
                    if isinstance(specie, Vacancy):
                        if vac_counted:
                            raise ValueError("More than one Vacancy species "
                                             "appear in a sub-lattice!")
                        comp_novac = Composition({k: v for k, v in comp.items()
                                                  if not isinstance(k, Vacancy)})
                        n.append((1 - comp_novac.num_atoms) * sl_size * sc_size)
                        vac_counted = True
                    else:
                        n.append(comp[specie] * sl_size * sc_size)
            n = np.array(n)
        else:
            raise ValueError(f"Composition format {form} not supported!")

        if rounding:
            n_round = np.array(np.round(n), dtype=int)
            if np.any(np.abs(n_round - n) > NUM_TOL):
                raise RoundingError(n, form="n")
            n = n_round.copy()

        return n

    def _convert_n_to(self, n, form, sc_size, rounding):
        n = np.array(n)
        if np.any(n < -NUM_TOL):
            raise NegativeSpeciesError(n, form="n")
        if np.any(np.abs(self.A @ (n / sc_size) - self.b)
                  > NUM_TOL):
            raise ConstraintViolationError(n, form="n")

        if form == "n":
            c = n.copy()
        elif form == "x":
            dn = n - self.n0 * sc_size
            c = np.linalg.pinv(self.basis.T) @ dn
        elif form == "comp":
            c = []
            for species, sl_size, dim_id in zip(self.bits,
                                                self.sl_sizes,
                                                self.dim_ids):
                n_sl = n[dim_id] / (sl_size * sc_size)
                c.append(Composition({sp: n for sp, n in zip(species, n_sl)
                                      if not isinstance(sp, Vacancy)}))
        elif form == "nondisc":
            c = np.zeros(len(self.species))
            for dim_id, dim_id_nondisc in zip(self.dim_ids,
                                              self.dim_ids_nondisc):
                c[dim_id_nondisc] += n[dim_id]
        else:
            raise ValueError(f"Composition format {form} not supported!")

        if rounding and form != "comp":
            c_round = np.array(np.round(c), dtype=int)
            if np.any(np.abs(c - c_round) > NUM_TOL):
                raise RoundingError(c, form=form)
            c = c_round.copy()
        return c

    def as_dict(self):
        """Serialize into dictionary.

        Return:
            Dict.
        """
        bits = [[sp.as_dict() for sp in sl_sps] for sl_sps in self.bits]

        n_cons = len(self.bits)
        if self.charge_balanced:
            n_cons += 1
        other_constraints = [(a, bb) for a, bb
                             in zip(self.A[n_cons:].tolist(),
                                    self.b[n_cons:].tolist())]
        comp_grids = {k: v.tolist() for k, v
                      in self._comp_grids.items()}

        def to_list(arr):
            if arr is not None:
                return np.array(arr).tolist()
            else:
                return None

        return {'bits': bits,
                'sl_sizes': self.sl_sizes,
                'other_constraints': other_constraints,
                'charge_balanced': self.charge_balanced,
                'optimize_basis': self.optimize_basis,
                'table_ergodic': self.table_ergodic,
                'min_sc_size': self._min_sc_size,
                'prim_vertices': to_list(self._prim_vertices),
                'n0': to_list(self._n0),
                'vs': to_list(self._vs),
                'flip_table': to_list(self._flip_table),
                'comp_grids': comp_grids,
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
        decoder = MontyDecoder()
        bits = [[decoder.process_decoded(sp_d) for sp_d in sl_sps]
                for sl_sps in d['bits']]
        sl_sizes = d.get('sl_sizes')
        other_constraints = d.get('other_constraints')
        charge_balanced = d.get('charge_balanced', True)
        optimize_basis = d.get('optimize_basis', False)
        table_ergodic = d.get('table_ergodic', False)

        obj = cls(bits, sl_sizes,
                  other_constraints=other_constraints,
                  charge_balanced=charge_balanced,
                  optimize_basis=optimize_basis,
                  table_ergodic=table_ergodic)

        obj._min_sc_size = d.get('min_sc_size')

        prim_vertices = d.get('prim_vertices')
        if prim_vertices is not None:
            obj._prim_vertices = np.array(prim_vertices)

        n0 = d.get('n0')
        if n0 is not None:
            obj._n0 = np.array(n0, dtype=int)

        vs = d.get('vs')
        if vs is not None:
            obj._vs = np.array(vs, dtype=int)

        flip_table = d.get('flip_table')
        if flip_table is not None:
            obj._flip_table = np.array(flip_table, dtype=int)

        comp_grids = d.get('comp_grids', {})
        comp_grids = {k: np.array(v, dtype=int)
                      for k, v in comp_grids.items()}
        obj._comp_grids = comp_grids

        return obj
