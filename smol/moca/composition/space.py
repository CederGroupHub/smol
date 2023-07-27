"""Defines a composition space with optional charge-neutrality or other constraints."""

__author__ = "Fengyu Xie"

import warnings
from itertools import chain

import numpy as np
from monty.json import MontyDecoder, MSONable
from pymatgen.core import Composition, Element

from smol.cofe.space.domain import Vacancy
from smol.moca.composition.constraints import CompositionConstraintsManager
from smol.moca.occu_utils import get_dim_ids_by_sublattice
from smol.utils.math import (
    NUM_TOL,
    get_ergodic_vectors,
    get_natural_centroid,
    get_natural_solutions,
    get_nonneg_float_vertices,
    get_optimal_basis,
    integerize_multiple,
    integerize_vector,
    solve_diophantines,
)


def get_oxi_state(sp):
    """Oxidation state from Specie/Element/Vacancy.

    Args:
       sp(Specie or Vacancy or Element):
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
        u(1D ArrayLike of int):
            The flip vector in number change of species.
        bits(list of lists of Specie or DummySpecie or Element or Vacancy):
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
                from_strs.append(f"{-u[dim]} {str(specie)}({sl_id})")
            elif u[dim] > 0:
                to_strs.append(f"{u[dim]} {str(specie)}({sl_id})")

    from_str = " + ".join(from_strs)
    to_str = " + ".join(to_strs)
    return from_str + " -> " + to_str


class CompositionSpace(MSONable):
    """Composition space class.

    Generates a compositional space with given charge neutrality or additional
    composition constraints and a given a list of Species or DummySpecies and the number
    of sites in each sub-lattice in a given unit cell.

    A composition in the CompositionSpace can be expressed in 4 formats:
        1, Count of species on each sub-lattice, ordered and concatenated as
           each sub-lattice provided in self.bits and self.sublattice_sizes,
           into an 1D array. ("counts" format, also called unconstrained
           coordinates)
        2, Coordinates x on the constrained integer grid, computed with
           n = n0 + V @ x from its counts format representation n,
           where V are the grid basi vectors and n0 is a base solution in
           a certain super-cell size. ("coordinates" format, also called
           constrained coordinates).
        3, Composition of each sub-lattice in pymatgen.Composition,
           but each sub-lattice composition is required to be divided
           by number of the sites in the sub-lattice, so that the sum of species
           content can not exceed 1. Vacancies contents are not explicitly
           included, but taken a 1 - sum(other_species_contents).
           ("compositions" format).
        4, Count each species in the whole structure, without distinguishing
           sub-lattice. The species are sorted, and the order of species is
           given in self.species. ("species-counts" format).

    Attributes:
        num_dims (int):
            The dimension of unconstrained composition space, which equals to
            the sum of site space size in all sub-lattices. Namely, it is the
            dimension of "counts" format.
        species (List of Species or DummySpecies or Element or Vacancy):
            All species in the given system (sorted).
        dim_ids (list of lists of int):
            The corresponding index in the "counts" vector
            of each species on each sub-lattice.
        species_ids (list of lists of int):
            The index of sorted species in each sub-lattice as shown in self.species.
    """

    other_constraints = CompositionConstraintsManager()

    def __init__(
        self,
        site_spaces,
        sublattice_sizes=None,
        charge_neutral=True,
        other_constraints=None,
        optimize_basis=False,
        table_ergodic=False,
    ):
        """Initialize CompositionSpace.

        Args:
            site_spaces (list of lists of Specie or Vacancy or Element):
                Site spaces specifying the allowed species on each sub-lattice.
            sublattice_sizes (1D ArrayLike of int): optional
                Number of sites in each sub-lattice per primitive cell.
                If not given, assume one site for each sub-lattice.
                Better provide them as co-prime integers.
            charge_neutral (bool): optional
                Whether to add charge balance constraint. Default is true.
            other_constraints
            (list of tuples of (1D arrayLike[float], float, str) or str): optional
                Other composition constraints to be applied to restrict the
                enumerated compositions.
                Allows two formats for each constraint in the list:
                    1, A string that encodes the constraint equation.
                    For example: "2 Ag+(0) + Cl-(1) +3 H+(2) <= 3 Mn2+ +4".
                       A string representation of constraint must satisfy the following
                       rules,
                       a, Contains a relation symbol ("==", "<=", ">=" or "=") are
                       allowed.
                       The relation symbol must have exactly one space before and one
                       space after to separate the left and the right sides.
                       b, Species strings must be readable by get_species in smol.cofe
                       .space.domain. No space is allowed within a species string.
                       For the format of a legal species string, refer to
                       pymatgen.core.species and smol.cofe.
                       c, You can add a number in brackets following a species string
                       to specify constraining the amount of species in a particular
                       sub-lattice. If not given, will apply the constraint to this
                       species on all sub-lattices.
                       This sub-lattice index label must not be separated from
                       the species string with space or any other character.
                       d, Species strings along with any sub-lattice index label must
                       be separated from other parts (such as operators and numbers)
                       with at least one space.
                       e, The intercept terms (a number with no species that follows)
                       must always be written at the end on both side of the equation.
                    2, The equation expression, which is a tuple containing a list of
                    floats of length self.n_dims to give the left-hand side coefficients
                    of each component in the composition "counts" format, a float to
                    give the right-hand side, and a string to specify the comparative
                    relationship between the left- and right-hand sides. Constrained in
                    the form of a_left @ n = (or <= or >=) b_right.
                    The components in the left-hand side are in the same order as in
                    itertools.chain(*self.bits).
                Note that all numerical values in the constraints must be set as they are
                to be satisfied per primitive cell given the sublattice_sizes!
                For example, if each primitive cell contains 1 site in 1 sub-lattice
                specified as sublattice_sizes=[1], with the requirement that species
                A, B and C sum up to occupy less than 0.6 sites per sub-lattice, then
                you must write: "A + B + C <= 0.6".
                While if you specify sublattice_sizes=[2] in the same system per
                primitive cell, to specify the same constraint, write
                "A + B + C <= 1.2" or "0.5 A + 0.5 B + 0.5 C <= 0.6", etc.
            optimize_basis (bool): optional
                Whether to optimize the basis to minimal flip sizes and maximal
                connectivity in the minimum super-cell size.
                When the minimal super-cell size is large, we recommend not to
                optimize basis.
            table_ergodic (bool): optional
                When generating a flip table, whether to add vectors and
                ensure ergodicity under a minimal super-cell size.
                Default to False.
                When the minimal super-cell size is large, we recommend not to
                ensure ergodicity. This is not only because of the computation
                difficulty; but also because at large super-cell size,
                the fraction of inaccessible compositions usually becomes
                minimal.
        """
        self.site_spaces = site_spaces
        self.num_dims = sum(len(species) for species in site_spaces)
        self.dim_ids = get_dim_ids_by_sublattice(self.site_spaces)
        # dimension of "n" format

        # For non-discriminative coordinates
        species = list(set(chain(*self.site_spaces)))
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

        species_ids = []
        for species in self.site_spaces:
            sl_dim_ids = []
            for sp in species:
                if not isinstance(sp, Vacancy):
                    sl_dim_ids.append(self.species.index(sp))
                else:
                    for sp2_id, sp2 in enumerate(self.species):
                        if isinstance(sp2, Vacancy):
                            sl_dim_ids.append(sp2_id)
                            break
            species_ids.append(sl_dim_ids)
        self.species_ids = species_ids

        if sublattice_sizes is None:
            self.sublattice_sizes = [1 for _ in range(len(self.site_spaces))]
        elif len(sublattice_sizes) == len(site_spaces):
            self.sublattice_sizes = np.array(sublattice_sizes, dtype=int).tolist()
        else:
            raise ValueError(
                "Sub-lattice number is not the same "
                "in parameters bits and sublattice_sizes."
            )

        self.charge_neutral = charge_neutral
        self.optimize_basis = optimize_basis
        self.table_ergodic = table_ergodic

        # Pre-process input constraints.
        self.other_constraints = other_constraints
        if self.other_constraints is not None:
            self._other_eq_constraints = self.other_constraints["eq"]
            self._other_leq_constraints = self.other_constraints["leq"]
        else:
            self._other_eq_constraints = []
            self._other_leq_constraints = []

        # Set constraint equations An=b (per primitive cell).
        A = []
        b = []
        if charge_neutral:
            A.append([get_oxi_state(sp) for species in site_spaces for sp in species])
            b.append(0)
        for dim_id, sublattice_size in zip(self.dim_ids, self.sublattice_sizes):
            a = np.zeros(self.num_dims, dtype=int)
            a[dim_id] = 1
            A.append(a.tolist())
            b.append(sublattice_size)
        for a, bb in self._other_eq_constraints:
            if len(a) != self.num_dims:
                raise ValueError(
                    f"Constraint length: {len(a)} does not match"
                    f" dimensions: {self.num_dims}!"
                )
            # No-longer enforce integers in a and b.
            # Integerize a.
            a_new, scale = integerize_vector(a)
            # Convert to array before multiplying!!
            A.append(np.round(np.array(a) * scale).astype(int))
            b.append(bb * scale)
        self._A = np.array(A, dtype=int)
        self._b = np.array(b)  # per-prim
        if np.linalg.matrix_rank(self._A) >= self.num_dims:
            raise ValueError("Valid constraints more than number of dimensions!")

        if len(self._other_leq_constraints) > 0:
            self._A_leq = np.array([a for a, bb in self._other_leq_constraints])
            self._b_leq = np.array([bb for a, bb in self._other_leq_constraints])
        else:
            self._A_leq = None
            self._b_leq = None

        self._prim_vertices = None
        # Minimum supercell size required to make prim_vertices coordinates
        # all integer.
        self._min_supercell_size = None
        self._flip_table = None
        self._n0 = None  # n0 base solution at minimum feasible sc size.
        self._vs = None  # Basis vectors, n = n0 + vs.T @ x
        self._comp_grids = {}  # Grid of integer compositions, in "x" format.

    @property
    def prim_vertices(self):
        """Vertex compositions in a primitive cell.

        leq constraints are not considered here.

        Returns:
            prim vertices in "counts" format:
                2D np.ndarray of float
        """
        if self._prim_vertices is None:
            self._prim_vertices = get_nonneg_float_vertices(self._A, self._b)
        return self._prim_vertices

    @property
    def min_supercell_size(self):
        """Minimum super-cell size.

        Computed as the minimum integer that can multiply prim_vertices into
        integral vectors.

        Returns: int
        """
        if self._min_supercell_size is None:
            int_verts, supercell_size = integerize_multiple(self.prim_vertices)
            self._min_supercell_size = supercell_size
        return self._min_supercell_size

    @property
    def num_unconstrained_compositions(self):
        """Estimated number of unconstrained compositions.

        Returns: int
        """
        return np.prod(
            [
                (sublattice_size * self.min_supercell_size) ** len(species)
                for species, sublattice_size in zip(
                    self.site_spaces, self.sublattice_sizes
                )
            ]
        )

    def get_supercell_base_solution(self, supercell_size=None):
        """Find one solution (not natural numbers) in a super-cell size.

        Args:
            supercell_size(int): optional
                Super-cell size in the number of primitive cells.
                If not given, will use self.min_supercell_size.
        Returns:
             1D np.ndarray of int
        """
        # This conserves scalability of grid enumeration, which means a grid
        # enumerated at supercell_size = 2 * a and step = 2 is exactly 2 * the
        # grid enumerated at supercell_size = a and step = 1. But make sure that
        # step = 1 case is called first.
        if supercell_size is None:
            supercell_size = self.min_supercell_size
        # minimum sc size that makes self._b an integer vector.
        # Any allowed supercell_size must be a multiple of it.
        _, min_feasible_supercell_size = integerize_vector(self._b)
        if not supercell_size % min_feasible_supercell_size == 0:
            raise ValueError(
                "Composition constraints can not have any integral "
                f"solution in a super-cell of {supercell_size} prims!"
            )
        if self._n0 is None:
            n0, vs = solve_diophantines(
                self._A, np.round(self._b * min_feasible_supercell_size)
            )
            # n0 and vs must always be ints.
            self._n0 = n0.copy()
        return self._n0 * supercell_size // min_feasible_supercell_size

    @property
    def basis(self):
        """Basis vectors of the integer composition grid.

        if self.optimize_basis is set to true, will be optimized so that flip
        sizes are minimal, and the graph connectivity between enumerated
        compositions in a self.min_supercell_size super-cell is maximized.
        Returns:
            Basis vectors in "counts" format, each in a row:
                2D np.ndarray of int
        """
        if self._vs is None:
            n0, vs = solve_diophantines(
                self._A, np.round(self._b * self.min_supercell_size)
            )
            if self.optimize_basis:
                n_comps = self.num_unconstrained_compositions
                if n_comps > 10**6:
                    warnings.warn(
                        "Basis optimization can be very costly "
                        "at your composition space size = "
                        f"{n_comps}. "
                        "Do this at your own risk!"
                    )
                xs = get_natural_solutions(n0, vs)
                vs_opt = get_optimal_basis(n0, vs, xs)
            else:
                vs_opt = vs.copy()
            self._vs = vs_opt
        return self._vs

    @property
    def flip_table(self):
        """Give flip table vectors.

        If self.table_ergodic is true, will add flips until the flip table is ergodic
        in a self.min_supercell_size super-cell.

        Returns: 2D np.ndarray of int
            Flip vectors in the "counts" format
        """
        if self._flip_table is None:
            if not self.table_ergodic:
                self._flip_table = self.basis.copy()
            else:
                n_comps = self.num_unconstrained_compositions
                if n_comps > 10**6:
                    warnings.warn(
                        "Ergodicity computation can be very costly "
                        "in your composition space, which has "
                        f"{n_comps} unconstrained compositions. "
                        "Do this at your own risk!"
                    )
                n0 = self.get_supercell_base_solution(self.min_supercell_size)
                self._flip_table = get_ergodic_vectors(
                    n0, self.basis, self.min_supercell_grid
                )
        return self._flip_table

    @property
    def flip_reactions(self):
        """Reaction formula representations of table flips.

        Return:
            All reaction formulas (only forward direction):
                list of str
        """
        return [flip_vec_to_reaction(u, self.site_spaces) for u in self.flip_table]

    def get_composition_grid(self, supercell_size=1, step=1):
        """Get the integer compositions ("coordinates").

        Args:
            supercell_size (int):
                Super-cell size to enumerate with.
            step (int): optional
                Step in returning the enumerated compositions.
                If step = N > 1, on each dimension of the composition space,
                we will only yield one composition every N compositions.
                Default to 1.

        Returns: 2D np.ndarray of int
            Integer compositions ("coordinates" format, not normalized):
        """
        # Also scalablity issue.
        scale = None
        supercell_size_prev = None
        step_prev = None
        for k1, k2 in self._comp_grids:
            if (
                supercell_size % k1 == 0
                and step % k2 == 0
                and supercell_size // k1 == step // k2
            ):
                scale = supercell_size // k1
                supercell_size_prev = k1
                step_prev = k2
                break

        if scale is not None:
            return self._comp_grids[(supercell_size_prev, step_prev)] * scale
        else:
            s = np.gcd(supercell_size, step)
            if s > 1:
                return (
                    self.get_composition_grid(
                        supercell_size=supercell_size // s, step=step // s
                    )
                    * s
                )
            else:
                n0 = self.get_supercell_base_solution(supercell_size)
                grid = get_natural_solutions(n0, self.basis, step=step)

                ns = grid @ self.basis + n0  # each row
                if self._A_leq is not None and self._b_leq is not None:
                    _filter_leq = (
                        self._A_leq @ ns.T / supercell_size
                        <= self._b_leq[:, None] + NUM_TOL
                    ).all(axis=0)
                else:
                    _filter_leq = np.ones(len(ns)).astype(bool)

                # Filter inequality constraints.
                self._comp_grids[(supercell_size, step)] = grid[_filter_leq]
                return self._comp_grids[(supercell_size, step)]

    @property
    def min_supercell_grid(self):
        """Get integer compositions on grid at min_supercell_size ("coordinates").

        Returns: 2D np.ndarray of int
            Integer compositions ("coordinates" format, not normalized):
        """
        return self.get_composition_grid(supercell_size=self.min_supercell_size)

    def get_centroid_composition(self, supercell_size=None):
        """Get the closest integer composition to the centroid of polytope.

        Args:
            supercell_size(int): optional
               Super-cell size to get the composition with.
               If not given, will use self.min_supercell_size
        Return: 1D np.ndarray of int
            the closest composition to the centroid ("coordinates" format, not normalized)
        """
        if supercell_size is None:
            supercell_size = self.min_supercell_size
        n0 = self.get_supercell_base_solution(supercell_size)
        return get_natural_centroid(
            n0,
            self.basis,
            supercell_size,
            self._A_leq,
            self._b_leq,
        )

    def translate_format(
        self, c, supercell_size, from_format, to_format="counts", rounding=False
    ):
        """Translate a composition representation to another format.

        Args:
            c (1D ArrayLike or list of Composition):
                Input format. Can be int or fractional.
            supercell_size (int):
                Supercell size of this composition. You must make sure
                it is the right super-cell size of composition c.
            from_format (str):
                Specifies the input format.
                A composition can be expressed in 4 formats:
                1, Count of species on each sub-lattice, ordered and concatenated as
                   each sub-lattice provided in self.bits and self.sublattice_sizes,
                   into an 1D array. ("counts" format)
                2, Coordinates x on the constrained integer grid, computed with
                   n = n0 + V @ x from its counts format representation n,
                   where V are the grid basi vectors and n0 is a base solution in
                   a certain super-cell size. ("coordinates" format).
                3, Composition of each sub-lattice in pymatgen.Composition,
                   but each sub-lattice composition is required to be divided
                   by number of the sites in the sub-lattice, so that the sum of species
                   content can not exceed 1. Vacancies contents are not explicitly
                   included, but taken a 1 - sum(other_species_contents).
                   ("compositions" format).
                4, Count each species in the whole structure, without distinguishing
                   sub-lattice. The species are sorted, and the order of species is
                   given in self.species. ("species-counts" format).
                Note: "species-counts" format cannot be converted to other formats,
                because it does not have information on species distribution across
                sub-lattices.
            to_format (str): optional
                Specified the output format.
                Same as from_format, with an addition "species-counts".
            rounding (bool): optional
                If the returned format is "counts", "coordinates" or "species-counts",
                whether to round up the output array as integers. Default to False.
        Return:
            Depends on the argument "to_format":
                1D np.ndarray of int or float, or list of Composition
        """
        if from_format == "species-counts":
            raise ValueError("species-counts can not be converted to other formats!")

        n = self._convert_to_counts(
            c, form=from_format, supercell_size=supercell_size, rounding=rounding
        )
        return self._convert_counts_to(
            n, form=to_format, supercell_size=supercell_size, rounding=rounding
        )

    def _convert_to_counts(self, c, form, supercell_size, rounding):
        """Convert other composition format to n-format."""
        if form == "counts":
            n = np.array(c)
        elif form == "coordinates":
            n = self.basis.transpose() @ np.array(c) + self.get_supercell_base_solution(
                supercell_size
            )
        elif form == "compositions":
            n = []
            for species, sublattice_size, comp in zip(
                self.site_spaces, self.sublattice_sizes, c
            ):
                if comp.num_atoms > 1 or comp.num_atoms < 0:
                    raise ValueError(
                        f"Sub-lattice composition {c} given in"
                        f" pymatgen.Composition format, but was"
                        f" not normalized to 1!"
                    )
                vac_counted = False
                for specie in species:
                    if isinstance(specie, Vacancy):
                        if vac_counted:
                            raise ValueError(
                                "More than one Vacancy species "
                                "appear in a sub-lattice!"
                            )
                        comp_novac = Composition(
                            {
                                k: v
                                for k, v in comp.items()
                                if not isinstance(k, Vacancy)
                            }
                        )
                        n.append(
                            (1 - comp_novac.num_atoms)
                            * sublattice_size
                            * supercell_size
                        )
                        vac_counted = True
                    else:
                        n.append(comp[specie] * sublattice_size * supercell_size)
            n = np.array(n)
        else:
            raise ValueError(f"Composition format {form} not supported!")

        if rounding:
            n_round = np.array(np.round(n), dtype=int)
            if np.any(np.abs(n_round - n) > NUM_TOL):
                raise ValueError(f"Composition {n} cannot be rounded into integers!")
            n = n_round.copy()

        return n

    def _convert_counts_to(self, n, form, supercell_size, rounding):
        n = np.array(n)
        if np.any(n < -NUM_TOL):
            raise ValueError(f"Composition {n} contains negative species count!")
        if np.any(np.abs(self._A @ (n / supercell_size) - self._b) > NUM_TOL):
            raise ValueError(
                f"Composition {n} violates constraints!"
                f" Numerical tolerance: {NUM_TOL}"
            )

        if form == "counts":
            c = n.copy()
        elif form == "coordinates":
            dn = n - self.get_supercell_base_solution(supercell_size)
            c = np.linalg.pinv(self.basis.T) @ dn
        elif form == "compositions":
            c = []
            for species, sublattice_size, dim_id in zip(
                self.site_spaces, self.sublattice_sizes, self.dim_ids
            ):
                n_sl = n[dim_id] / (sublattice_size * supercell_size)
                c.append(
                    Composition(
                        {
                            sp: n
                            for sp, n in zip(species, n_sl)
                            if not isinstance(sp, Vacancy)
                        }
                    )
                )
        elif form == "species-counts":
            c = np.zeros(len(self.species))
            for dim_id, species_ids in zip(self.dim_ids, self.species_ids):
                c[species_ids] += n[dim_id]
        else:
            raise ValueError(f"Composition format {form} not supported!")

        if rounding and form != "compositions":
            c_round = np.array(np.round(c), dtype=int)
            if np.any(np.abs(c - c_round) > NUM_TOL):
                raise ValueError(f"Composition {c} cannot be rounded into integers!")
            c = c_round.copy()
        return c

    def as_dict(self):
        """Serialize into dictionary.

        Return:
            dict
        """
        bits = [[sp.as_dict() for sp in sl_sps] for sl_sps in self.site_spaces]

        n_cons = len(self.site_spaces)
        if self.charge_neutral:
            n_cons += 1
        eq_constraints = [
            (a, bb, "eq")
            for a, bb in zip(self._A[n_cons:].tolist(), self._b[n_cons:].tolist())
        ]
        leq_constraints = (
            [
                (a, bb, "leq")
                for a, bb in zip(self._A_leq.tolist(), self._b_leq.tolist())
            ]
            if self._A_leq is not None and self._b_leq is not None
            else []
        )

        other_constraints = eq_constraints + leq_constraints

        comp_grids = {
            f"{k[0]}_{k[1]}": v.tolist() for k, v in self._comp_grids.items()
        }  # Encode tuple keys.

        def to_list(arr):
            if arr is not None:
                return np.array(arr).tolist()
            else:
                return None

        return {
            "bits": bits,
            "sublattice_sizes": self.sublattice_sizes,
            "other_constraints": other_constraints,
            "charge_neutral": self.charge_neutral,
            "optimize_basis": self.optimize_basis,
            "table_ergodic": self.table_ergodic,
            "min_supercell_size": self._min_supercell_size,
            "prim_vertices": to_list(self._prim_vertices),
            "n0": to_list(self._n0),
            "vs": to_list(self._vs),
            "flip_table": to_list(self._flip_table),
            "comp_grids": comp_grids,
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
        }

    @classmethod
    def from_dict(cls, d):
        """Load CompositionSpace object from dictionary.

        Args:
            d(dict):
                Dictionary to decode from.
        Return:
            CompositionSpace
        """
        decoder = MontyDecoder()
        bits = [
            [decoder.process_decoded(sp_d) for sp_d in sl_sps] for sl_sps in d["bits"]
        ]
        sublattice_sizes = d.get("sublattice_sizes")
        other_constraints = d.get("other_constraints")
        charge_neutral = d.get("charge_neutral", True)
        optimize_basis = d.get("optimize_basis", False)
        table_ergodic = d.get("table_ergodic", False)

        obj = cls(
            bits,
            sublattice_sizes,
            other_constraints=other_constraints,
            charge_neutral=charge_neutral,
            optimize_basis=optimize_basis,
            table_ergodic=table_ergodic,
        )

        obj._min_supercell_size = d.get("min_supercell_size")

        prim_vertices = d.get("prim_vertices")
        if prim_vertices is not None:
            obj._prim_vertices = np.array(prim_vertices)

        n0 = d.get("n0")
        obj._n0 = np.array(n0, dtype=int) if n0 is not None else None

        vs = d.get("vs")
        if vs is not None:
            obj._vs = np.array(vs, dtype=int)

        flip_table = d.get("flip_table")
        if flip_table is not None:
            obj._flip_table = np.array(flip_table, dtype=int)

        comp_grids = d.get("comp_grids", {})
        comp_grids = {
            (int(k.split("_")[0]), int(k.split("_")[1])): np.array(v, dtype=int)
            for k, v in comp_grids.items()
        }  # Decode tuple keys.
        obj._comp_grids = comp_grids

        return obj
