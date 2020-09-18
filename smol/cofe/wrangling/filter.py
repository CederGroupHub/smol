"""Functions to filter data in a StructureWrangler."""

from collections import defaultdict
from functools import wraps
import numpy as np
from smol.cofe.extern import EwaldTerm

__author__ = "Luis Barroso-Luque, William Davidson Richard"


def register_filter_metadata(filter_fun):
    """Decorator to register metadata associated with a filter.

    Decorate filter functions with this to register the metadata associated
    with the filtering.
    Args:
        filter_fun (Callable):
            A filter function that takes a StructureWrangler as the first
            argument (and any necessary additional keyword arguments) to
            filter data items in the StructureWrangler.
    Returns:
        Callable: decorated filter function
    """

    @wraps(filter_fun)
    def reg_filter(wrangler, **kwargs):
        metadata = {"initial_num_items": len(wrangler.data_items)}
        metadata.update(kwargs)
        filter_fun(wrangler, **kwargs)
        metadata["final_num_items"] = len(wrangler.data_items)
        wrangler.metadata["applied_filters"].append(
            {filter_fun.__name__: metadata})

    return reg_filter


@register_filter_metadata
def filter_duplicate_corr_vectors(wrangler, property_key, filter_by='min',
                                  verbose=False):
    """Filter structures with duplicate correlation vectors.

    Keep structures with the min or max value of the given property.

    Args:
        wrangler (StructureWrangler):
            A StructureWrangler containing data to be filtered.
        property_key (str):
            Name of property to consider when returning structure indices
            for the feature matrix with duplicate corr vectors removed.
        filter_by (str)
            The criteria for the property value to keep. Options are min
            or max.
        verbose (bool): optional
            If True print the filtered structures.
    """
    if filter_by not in ('max', 'min'):
        raise ValueError(f'Filtering by {filter_by} is not an option.')

    choose_val = np.argmin if filter_by == 'min' else np.argmax

    property_vector = wrangler.get_property_vector(property_key)
    dupe_inds = wrangler.get_duplicate_corr_inds()
    to_keep = {inds[choose_val(property_vector[inds])]
               for inds in dupe_inds}
    to_remove = set(i for inds in dupe_inds for i in inds) - to_keep
    if verbose:
        for i in to_remove:
            item = wrangler.data_items[i]
            print(f"Removing structure {item['structure'].composition}."
                  f"with properties {item['properties']}.")
    data_items = [item for i, item in enumerate(wrangler.data_items)
                  if i not in to_remove]
    wrangler.remove_all_data()
    wrangler.append_data_items(data_items)


@register_filter_metadata
def filter_data_by_ewald(wrangler, max_ewald, verbose=False):
    """Filter structures by electrostatic interaction energy.

    Filter the input structures to only use those with low electrostatic
    energies (no large charge separation in cell). This energy is
    referenced to the lowest value at that composition. Note that this is
    before the division by the relative dielectric constant and is per
    primitive cell in the cluster expansion -- 1.5 eV/atom seems to be a
    reasonable value for dielectric constants around 10.

    Args:
        wrangler (StructureWrangler):
            A StructureWrangler containing data to be filtered.
        max_ewald (float):
            Ewald threshold. The maximum Ewald energy, normalized by prim
        verbose (bool): optional
            If True will print filtered structures.
    """
    ewald_energy = None
    for term in wrangler.cluster_subspace.external_terms:
        if isinstance(term, EwaldTerm):
            ewald_energy = [i['features'][-1] for i in wrangler.data_items]
    if ewald_energy is None:
        ewald_energy = []
        for item in wrangler.data_items:
            struct = item['structure']
            matrix = item['scmatrix']
            mapp = item['mapping']
            occu = wrangler.cluster_subspace.occupancy_from_structure(
                struct,
                encode=True,
                scmatrix=matrix,
                site_mapping=mapp)

            supercell = wrangler.cluster_subspace.structure.copy()
            supercell.make_supercell(matrix)
            size = wrangler.cluster_subspace.num_prims_from_matrix(matrix)
            term = EwaldTerm().value_from_occupancy(occu, supercell) / size
            ewald_energy.append(term)

    min_energy_by_comp = defaultdict(lambda: np.inf)
    for energy, item in zip(ewald_energy, wrangler.data_items):
        c = item['structure'].composition.reduced_composition
        if energy < min_energy_by_comp[c]:
            min_energy_by_comp[c] = energy

    data_items = []
    for energy, item in zip(ewald_energy, wrangler.data_items):
        min_energy = min_energy_by_comp[
            item['structure'].composition.reduced_composition]
        rel_energy = energy - min_energy
        if rel_energy > max_ewald and verbose:
            print(f"Removing structure {item['structure'].composition} "
                  f"with properties {item['properties']}.\n"
                  f"Relative Ewald interaction energy is {rel_energy} eV.")
        else:
            data_items.append(item)
    wrangler.remove_all_data()
    wrangler.append_data_items(data_items)
