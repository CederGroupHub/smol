import numpy as np
from collections import defaultdict
from pymatgen import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry

def _pd(structures, energies, ce):
    """
    Generate a phase diagram with the structures and energies
    """
    entries = []

    for s, e in zip(structures, energies):
        entries.append(PDEntry(s.composition.element_composition, e))

    max_e = max(entries, key=lambda e: e.energy_per_atom).energy_per_atom + 1000
    for el in ce.structure.composition.keys():
        entries.append(PDEntry(Composition({el: 1}).element_composition, max_e))

    return PhaseDiagram(entries)


def _energies_above_composition(structures, energies):
    min_e = defaultdict(lambda: np.inf)
    for s, e in zip(structures, energies):
        comp = s.composition.reduced_composition
        if e / len(s) < min_e[comp]:
            min_e[comp] = e / len(s)
    e_above = []
    for s, e in zip(structures, energies):
        comp = s.composition.reduced_composition
        e_above.append(e / len(s) - min_e[comp])
    return np.array(e_above)


def _energies_above_hull(pd, structures, energies):
    e_above_hull = []
    for s, e in zip(structures, energies):
        e_above_hull.append(pd.get_e_above_hull(PDEntry(s.composition.element_composition, e)))
    return np.array(e_above_hull)


def weight_by_e_above_hull(cls, cluster_expansion, structures, energies, mu=None, max_dielectric=None,
                           max_ewald=None, temperature=2000, solver='cvxopt_l1'):
    pd = _pd(structures, energies, cluster_expansion)
    e_above_hull = _energies_above_hull(pd, structures, energies)
    weights = np.exp(-e_above_hull / (0.00008617 * temperature))

    return cls(cluster_expansion=cluster_expansion, structures=structures, energies=energies,
               mu=mu, weights=weights, max_dielectric=max_dielectric, max_ewald=max_ewald, solver=solver)


def weight_by_e_above_comp(cls, cluster_expansion, structures, energies, mu=None, max_dielectric=None,
                           max_ewald=None, temperature=2000, solver='cvxopt_l1'):
    e_above_comp = _energies_above_composition(structures, energies)
    weights = np.exp(-e_above_comp / (0.00008617 * temperature))

    return cls(cluster_expansion=cluster_expansion, structures=structures, energies=energies,
               mu=mu, weights=weights, max_dielectric=max_dielectric, max_ewald=max_ewald, solver=solver)
