"""Utility manager that converts input of composition constraints to equations."""
import re
from collections import Counter
from numbers import Number

import numpy as np

from smol.cofe.space.domain import get_species
from smol.moca.utils.occu import get_dim_ids_by_sublattice


def handle_side_string(side: str):
    """Handle the strings separated by relation operators.

    Args:
        side(str):
           A side of equation in string.
    Returns:
        list[tuple(Number, Species, int|None)], Number:
            A list of tuples packing coefficients, species, and sub-lattice indices (if any)
            on this side of equation, and the intercept term.
    """
    # split around spaces. each literal can be either "+", "-", "\d" or species string.
    literals = side.split()
    processed_literals = []

    numerical_reg = re.compile(r"^[+-]?\d+\.?\d*$")
    sublattice_reg = re.compile(r"^([A-Za-z]+.*)\((\d+)\)$")
    for i, lit in enumerate(literals):
        if lit == "+" or lit == "-":
            processed_literals.append(lit)
        elif numerical_reg.match(lit):
            # might be a float.
            if "." in lit:
                num = float(lit)
            # Integer
            else:
                num = int(lit)
            # process to int if possible.
            if np.isclose(num, round(num)):
                num = round(num)
            processed_literals.append(num)
        else:
            sublattice_match = sublattice_reg.match(lit)
            # Specified a sub-lattice index.
            if sublattice_match is not None:
                species = get_species(sublattice_match.groups()[0])
                sublattice_id = int(sublattice_match.groups()[1])
            # No sub-lattice index.
            else:
                species = get_species(lit)
                sublattice_id = None
            processed_literals.append((species, sublattice_id))

    def extract_symbols_before_number(all_literals, number_literal_id):
        num_symbols = 0
        symbol = 1
        if number_literal_id > 0:
            for jj in range(1, number_literal_id + 1):
                if all_literals[number_literal_id - jj] == "+":
                    num_symbols += 1
                elif all_literals[number_literal_id - jj] == "-":
                    symbol = -1 * symbol
                    num_symbols += 1
                else:
                    break

        return symbol, num_symbols

    def extract_number_before_species(all_literals, species_literal_id):
        num_numbers = 0
        if species_literal_id > 0:
            for jj in range(1, species_literal_id + 1):
                if isinstance(all_literals[species_literal_id - jj], Number):
                    num_numbers += 1
                else:
                    break
        if num_numbers == 0:
            number_literal = 1
        elif num_numbers == 1:
            number_literal = all_literals[species_literal_id - 1]
        else:
            raise ValueError(
                f"Species {all_literals[species_literal_id]} preceded by"
                f" {num_numbers} > 1"
                f" literals, not allowed!"
            )

        return number_literal, num_numbers

    # Pack literals and intercept term.
    intercept = 0
    if isinstance(processed_literals[-1], Number):
        sym, n_sym = extract_symbols_before_number(
            processed_literals, len(processed_literals) - 1
        )
        intercept = sym * processed_literals[-1]
        processed_literals = processed_literals[: -(1 + n_sym)]
    elif processed_literals[-1] == "+":
        intercept = 1
        processed_literals = processed_literals[:-1]
    elif processed_literals[-1] == "-":
        intercept = -1
        processed_literals = processed_literals[:-1]

    packed_literals = []
    # A species literal should never be preceded by more than 1 number literals.
    for i, lit in enumerate(processed_literals):
        if isinstance(lit, tuple):
            # Extract the number literal before a species literal.
            num, n_nums = extract_number_before_species(processed_literals, i)
            sym, n_syms = extract_symbols_before_number(processed_literals, i - n_nums)

            packed_literals.append((sym * num, *lit))

    return packed_literals, intercept


def convert_constraint_string(entry, bits):
    """Convert a string containing a composition constraint to standard form.

    Args:
        entry(str):
           A string containing a constraint to composition. For example:
               "2 Ag+(0) + Cl-(1) +3 H+(2) <= 3 Mn2+ +4"
           There are a few requirements for writing string:
           1, Three relations, "==", "<=", ">=", are allowed. These symbols
              separate the left and the right side. They must be wrapped by
              exactly one space before and one space after.
           2, You can include a number in brackets following a species
              string to constrain the amount of species in a particular
              sub-lattice. If no sub-lattice index is given, will apply
              the constraint to this particular species on all sub-lattices.
           3, Species strings (with sub-lattice index in the bracket, if any)
              must be separated from its preceding number coefficient by a
              single space.
           4, Species strings must be separated from other contents before
              and after them by at least one space, but no space is allowed
              within a species string, nor between a species string and its
              sub-lattice index specifier.
              For the format of species strings, refer to pymatgen.core.species.
           5, Intercept terms must be written at the end of each side of the
              equation.
           6, All coefficients and number terms in the equation must be set
              according to the sub-lattice sizes in CompositionSpace object.
        bits(list[list[Species|Vacancy|Element]]):
            Allowed species on each sub-lattice. Must be in the same ordering
            as will appear in moca.composition.
    Returns:
        list[Number], Number, str:
            Left side of the equation, right side of the equation (simplified
            to only contain a number), and the label of relation. All in "counts"
            format as specified in moca.composition.
    """
    entry = entry.strip()
    separator_reg = re.compile(r"^(.*) ([<=>]?=) (.*)$")
    sep = separator_reg.match(entry)
    if sep is None:
        raise ValueError(
            f"Provided string entry {entry} must have <=, >=, == or"
            " = symbol wrapped by two spaces in the middle!"
        )
    left_string, relation, right_string = sep.groups()

    # Separate left and right side into terms containing species or numbers.
    left_pack, left_intercept = handle_side_string(left_string)
    right_pack, right_intercept = handle_side_string(right_string)

    n_dims = sum([len(species) for species in bits])
    dim_ids = get_dim_ids_by_sublattice(bits)
    left_vec = [0 for _ in range(n_dims)]

    # Add left.
    for coef, spec, sl_id in left_string:
        # Specified to constrain the species on only one sub-lattice.
        if sl_id is not None:
            dim_id = dim_ids[sl_id][bits[sl_id].index(spec)]
            left_vec[dim_id] += coef
        # The corresponding species on all sub-lattices should be constrained.
        else:
            for species, sub_dim_ids in zip(bits, dim_ids):
                if spec in species:
                    dim_id = sub_dim_ids[species.index(spec)]
                    left_vec[dim_id] += coef

    # Subtract right.
    for coef, spec, sl_id in right_string:
        # Specified to constrain the species on only one sub-lattice.
        if sl_id is not None:
            dim_id = dim_ids[sl_id][bits[sl_id].index(spec)]
            left_vec[dim_id] -= coef
        # The corresponding species on all sub-lattices should be constrained.
        else:
            for species, sub_dim_ids in zip(bits, dim_ids):
                if spec in species:
                    dim_id = sub_dim_ids[species.index(spec)]
                    left_vec[dim_id] -= coef

    right = right_intercept - left_intercept

    return left_vec, right, relation


class CompositionConstraintsManager:
    """A descriptor class that manages setting composition constraints."""

    def __set_name__(self, owner, name):
        """Set the private variable names."""
        self.public_name = name
        self.private_name = "_" + name

    def __get__(self, obj, objtype=None):
        """Return the chemical potentials if set None otherwise."""
        value = getattr(obj, self.private_name, None)
        return value if value is None else value["value"]

    @staticmethod
    def _check_single_dict(d):
        for spec, count in Counter(map(get_species, d.keys())).items():
            if count > 1:
                raise ValueError(
                    f"{count} values of the constraint coefficient for the same "
                    f"species {spec} were provided.\n Make sure the dictionary "
                    "you are using has only string keys or only Species "
                    "objects as keys."
                )

    @staticmethod
    def _convert_single_dict(left, bits):
        # Set a constraint with only one dictionary.
        CompositionConstraintsManager._check_single_dict(left)
        n_dims = sum([len(sublattice_bits) for sublattice_bits in bits])
        dim_ids = get_dim_ids_by_sublattice(bits)
        left_list = [0 for _ in range(n_dims)]
        for spec, coef in left.items():
            spec = get_species(spec)
            for sl_dim_ids, sl_bits in zip(dim_ids, bits):
                dim_id = sl_dim_ids[sl_bits.index(spec)]
                left_list[dim_id] = coef
        return left_list

    @staticmethod
    def _convert_sublattice_dicts(left, bits):
        # Set a constraint with one dict per sub-lattice.
        n_dims = sum([len(sublattice_bits) for sublattice_bits in bits])
        dim_ids = get_dim_ids_by_sublattice(bits)
        left_list = [0 for _ in range(n_dims)]
        for sl_dict, sl_bits, sl_dim_ids in zip(left, bits, dim_ids):
            CompositionConstraintsManager._check_single_dict(sl_dict)
            for spec, coef in sl_dict.items():
                spec = get_species(spec)
                dim_id = sl_dim_ids[sl_bits.index(spec)]
                left_list[dim_id] = coef
        return left_list

    def __set__(self, obj, value):
        """Set the table given the owner and value."""
        if value is None:  # call delete if set to None
            self.__delete__(obj)
            return

        # value must be list of tuples, each with a list and a number.
        # No scaling would be done. Take care when filling in!
        constraints_eq = []
        constraints_leq = []
        constraints_geq = []
        bits = [sublattice.species for sublattice in obj.sublattices]

        def _contains_dict(item):
            if isinstance(item, dict):
                return True
            elif hasattr(item, "__iter__"):
                return any(_contains_dict(sub_item) for sub_item in item)
            return False

        for entry in value:
            if isinstance(entry, (tuple, list)):
                if _contains_dict(entry):
                    left, right, relation = entry
                    # Constraint specified as non lattice-specific dictionary.
                    if isinstance(left, dict):
                        left_vec = self._convert_single_dict(left, bits)
                    # Constraint specified as lattice-specific dictionaries.
                    else:
                        left_vec = self._convert_sublattice_dicts(left, bits)
                # Constraints specified directly as left, right, relation.
                else:
                    left_vec, right, relation = entry
            # Constraint set as a string.
            elif isinstance(entry, str):
                left_vec, right, relation = convert_constraint_string(entry, bits)
            else:
                raise ValueError(
                    f"Constraint {entry} format is not readable. See"
                    f" smol.moca.composition for detailed description of formats."
                )

            if relation == "leq" or relation == "<=":
                constraints_leq.append((left_vec, right))
            if relation == "geq" or relation == ">=":
                constraints_geq.append((left_vec, right))
            if relation == "eq" or relation == "==" or relation == "=":
                constraints_eq.append((left_vec, right))

        value = {
            "leq": constraints_leq,
            "geq": constraints_geq,
            "eq": constraints_eq,
        }

        # if first instantiation concatenate the natural parameter
        if not hasattr(obj, self.private_name):
            setattr(
                obj,
                self.private_name,
                {"value": value},
            )

    def __delete__(self, obj):
        """Delete the boundary condition."""
        if hasattr(obj, self.private_name):
            del obj.__dict__[self.private_name]