"""Utility manager that converts input of composition constraints to equations."""
import re
from collections import Counter

from smol.cofe.space.domain import get_species
from smol.moca.utils.occu import get_dim_ids_by_sublattice


def convert_constraint_string(entry):
    """Convert a string containing a composition constraint to standard form.

    Args:
        entry(str):
           A string containing a constraint to composition. For example:
               "2 Ag+(0) + Cl-(1) +3 H+(2) <= 3 Mn2+ +4"
           There are a few requirements for writing string:
           1, Three relations, "==", "<=", ">=", are allowed. These symbols
              separate the left and the right side. They and other operator
              symbols ("+", "-") must be wrapped by exactly one space before
              and one space after.
           2, You can include a number in brackets following a species
              string to constrain the amount of species in a particular
              sub-lattice. If no sub-lattice index is given, will apply
              the constraint to this particular species on all sub-lattices.
           3, Species strings (with sub-lattice index in the bracket, if any)
              must be separated from its precedding number coefficient by a
              single space.
              For the format of species strings, refer to pymatgen.core.species.
           4, All coefficients and intercepts in the equation must be set
              according to the sub-lattice sizes in CompositionSpace object.
    Returns:
        list[float], float, str:
            Left side of the equation, right side of the equation (simplified
            to only contain a number), and the label of relation.
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
        if value is None or len(value) == 0:  # call delete if set to None
            self.__delete__(obj)
            return

        # value must be list of tuples, each with a list and a number.
        # No scaling would be done. Take care when filling in!
        a_matrix_eq = []
        b_array_eq = []
        a_matrix_leq = []
        b_array_leq = []
        a_matrix_geq = []
        b_array_geq = []
        bits = [sublattice.species for sublattice in obj.sublattices]
        for entry in value:
            if isinstance(entry, (tuple, list)):
                left, right, relation = entry
                if isinstance(left, dict):
                    left_vec = self._convert_single_dict(left, bits)
                else:
                    left_vec = self._convert_sublattice_dicts(left, bits)
            elif isinstance(entry, str):
                left_vec, right, relation = convert_constraint_string(entry)
            else:
                raise ValueError(
                    f"Constraint {entry} must either be given in a string"
                    f" or given as a list."
                )

            if relation == "leq" or relation == "<=":
                a_matrix_leq.append(left_vec)
                b_array_leq.append(right)
            if relation == "geq" or relation == ">=":
                a_matrix_geq.append(left_vec)
                b_array_geq.append(right)
            if relation == "eq" or relation == "==" or relation == "=":
                a_matrix_eq.append(left_vec)
                b_array_eq.append(right)

        value = {
            "leq": (a_matrix_leq, b_array_leq),
            "geq": (a_matrix_geq, b_array_geq),
            "eq": (a_matrix_eq, b_array_eq),
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
