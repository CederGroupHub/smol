"""Get variables from processor."""
from typing import List, Tuple

import cvxpy as cp
import numpy as np

from smol.moca.processor.base import Processor

__author__ = "Fengyu Xie"


def get_upper_bound_variables_from_processor(
    processor: Processor,
) -> Tuple[cp.Variable, List[List[int]]]:
    """Get cvxpy boolean variables for the upper-bound problem from processor.

    Inactive sites (sites will only 1 allowed species) will not be added into variables.
    Args:
        processor(Processor):
            A processor to generate variables for.
    Returns:
        cp.Variable, list[list[int]]:
          Flatten variables for each active site and species; list of variable
          index corresponding to each active site index and species indices in
          its site space.
    """
    n_variables = 0
    variable_indices = []
    for species in processor.allowed_species:
        if len(species) > 1:
            variable_indices.append(
                list(range(n_variables, n_variables + len(species)))
            )
            n_variables += len(species)
        else:
            variable_indices.append([])
    return cp.Variable(n_variables, boolean=True), variable_indices


def get_occupancy_from_variables(
    variables: cp.Variable, variable_indices: List[List[int]]
) -> np.ndarray:
    """Get encoded occupancy array from variables.

    Args:
        variables(cp.Variable):
            cvxpy variables storing the ground-state result.
        variable_indices(list[list[int]]):
            List of variable indices corresponding to each active site index and
            species indices in its site space.
    Returns:
        np.ndarray: Encoded occupancy string.
    """
    if variables.value is None:
        raise ValueError("CVX variables are not solved yet!")
    occu = np.zeros(len(variable_indices), dtype=int) - 1
    # Not considering species encoding order yet.
    for site_id, indices in enumerate(variable_indices):
        if len(indices) > 0:
            species_ids_on_site = np.where(variables.value[indices])[0]
            if len(species_ids_on_site) > 1:
                raise ValueError(f"More than one species occupied site {site_id}!")
            occu[site_id] = species_ids_on_site[0]
        else:
            occu[site_id] = 0
    if not np.all(occu >= 0):
        raise ValueError(f"Variables does not match given indices: {variable_indices}!")

    return occu
