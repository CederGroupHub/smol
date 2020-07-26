#!/usr/bin/env python

__author__ = "Bin Ouyang"

import os
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher, OrderDisorderElementComparator


class PoscarWithChgDecorator(Poscar):
    """
    Allow process with charge decorator in POSCAR
    """

    def __init__(self, structure, comment=None, selective_dynamics=None, true_names=True, velocities=None,
                 predictor_corrector=None, predictor_corrector_preamble=None):
        """
        Esentially the same as Poscar class but allow the implement of ChargedStructure
        """
        if structure.is_ordered:
            site_properties = {}
            if selective_dynamics:
                site_properties["selective_dynamics"] = selective_dynamics
            if velocities:
                site_properties["velocities"] = velocities
            if predictor_corrector:
                site_properties["predictor_corrector"] = predictor_corrector
            self.structure = structure.copy()
            self.true_names = true_names
            self.comment = structure.formula if comment is None else comment
            self.predictor_corrector_preamble = predictor_corrector_preamble
        else:
            raise ValueError("Structure with partial occupancies cannot be "
                             "converted into POSCAR!")

        self.temperature = -1

    @staticmethod
    def from_file(filename, check_for_POTCAR=True, read_velocities=True):
        """
        Read decorator from file
        """
        old_pos_obj = Poscar.from_file(filename, check_for_POTCAR=check_for_POTCAR,
                                       read_velocities=read_velocities)
        oldstructure = old_pos_obj.structure
        elements = list(oldstructure .symbol_set)
        natoms = len(oldstructure)
        chglst = []
        with open(filename, 'r') as fid:
            countl = 1
            while 1:
                line = fid.readline()
                # So go to the last line that is not coordinates. The idea is to find the line
                # with Direct or Cartesian notes. However need to distinguish that from
                # elements line and selective dynamics line. May miss some invalid structures
                if (countl > 6) and ('d' in line or 'D' in line or 'C' in line or 'c' in line) \
                        and ('elective' not in line):
                    break
                else:
                    countl += 1
            for atomind in range(natoms):
                ion = fid.readline().split(' ')[-1]
                for ele in elements:
                    if ele in ion:
                        if ele == 'F' and 'Fe' in ion:
                            continue  # Get rid of the confusing of Fe as F
                        decorator = ion.replace(ele, '')
                        decorator = decorator.split()[0]
                        if len(decorator) == 1:
                            if decorator == '+':
                                chglst.append(1)
                            elif decorator == '-':
                                chglst.append(-1)
                            else:
                                print('Check Decorator {}'.format(decorator))
                        elif len(decorator) == 0:
                            chglst.append(0)
                        else:
                            if '+' in decorator:
                                chglst.append(eval(decorator.replace('+', '')))
                            elif '-' in decorator:
                                chglst.append(-1 * eval(decorator.replace('-', '')))
                            else:
                                print('Check Decorator {}'.format(decorator))
        oldstructure.add_oxidation_state_by_site(chglst)
        chgstr = ChargedStructure.from_structure(oldstructure)

        return PoscarWithChgDecorator(chgstr, old_pos_obj.comment, old_pos_obj.selective_dynamics,
                                      old_pos_obj.true_names, old_pos_obj.velocities, old_pos_obj.predictor_corrector,
                                      old_pos_obj.predictor_corrector_preamble)


class ChargedStructure(Structure):
    """
    Allow more charge operations for the structure object
    """

    def __init__(self, lattice: Lattice, species: list, coords: np.ndarray, charge: float = None,
                 validate_proximity: bool = False, to_unit_cell: bool = False, coords_are_cartesian: bool = False,
                 site_properties: dict = None):
        super(ChargedStructure, self).__init__(lattice, species, coords, charge, validate_proximity, to_unit_cell,
                                               coords_are_cartesian, site_properties)

    def get_ion_indices(self, ion):
        """
        Get the indices of certain ion, return a list
        """
        IndLst = []
        for Ind, Site in enumerate(self.sites):
            if str(Site.specie) == ion:
                IndLst.append(Ind)
        return IndLst

    @property
    def ion_symbols(self):
        """
        Get the symbols of all ions
        """
        return list(self.composition.as_dict().keys())

    @property
    def cation_symbols(self):
        """
        Get the symbols of cations
        """
        cations = []
        for ion, num in self.composition.items():
            if ion.oxi_state > 0:
                cations.append(ion)
        return cations

    @property
    def anion_symbols(self):
        """
        Get the symbols of anions
        """
        anions = []
        for ion, num in self.composition.items():
            if ion.oxi_state > 0:
                anion.append(ion)
        return anions

    @staticmethod
    def from_structure(structure):
        return ChargedStructure(structure.lattice, structure.species, structure.frac_coords,
                                structure.charge, validate_proximity=True, to_unit_cell=False,
                                coords_are_cartesian=False, site_properties=structure.site_properties)


def refine_str_to_reference(structure, refstr, stol, ltol, angle_tol):
    """
    Map the structure to a reference structure, also return a refined structure as a reference

    Str: The structure to be remapped
    RefStr: The "template" strucutre (usually have partial occupancy)

    returns:
        RMSE: Root mean square error of the match from structurematcher
        DispLst: A list of displacement of each site from structurematcher
        RemappedStr: Remapped Structure
    """
    # The matcher to get supercell matrix
    sm_scmat = StructureMatcher(primitive_cell=False, attempt_supercell=True,
                                allow_subset=True, scale=True, supercell_size='num_sites',
                                comparator=OrderDisorderElementComparator(),
                                stol=stol, ltol=ltol, angle_tol=angle_tol)
    # The matcher to map different sites
    sm_sites = StructureMatcher(primitive_cell=False, attempt_supercell=False,
                                allow_subset=True, scale=True, supercell_size='num_sites',
                                comparator=OrderDisorderElementComparator(),
                                stol=stol, ltol=ltol, angle_tol=angle_tol)
    # This is a generic implementation in case the unrelaxed structure is missing,
    # usually the scmat can be obtained from mapping the unrelaxed structure to CE
    # prim structure as well:
    scmat = sm_scmat.get_supercell_matrix(structure, refstr)
    rescaledstr = refstr.copy()
    rescaledstr.make_supercell(scmat)
    # In the below case, the mappings will be be a list of indices.
    # The index of each index in the list refers to the indice of site in Str.
    # The index in the list refers to the indice of site in RescaledStr
    # We need more information here so do not use get_mapping method in structurematcher
    # Mappings=sm_sites.get_mapping(RescaledStr,Str).tolist();
    superset, subset, _, _ = sm_sites._preprocess(rescaledstr, structure, True)
    match = sm_sites._strict_match(superset, subset, 1, break_on_match=False)
    # match == val, dist, sc_m, total_t, mapping
    rmse, displst, mappings = match[0], match[1], match[4]
    splst = []
    coordlst = []
    for i_str, i_refstr in enumerate(mappings):
        splst.append(structure[i_str].specie)
        coordlst.append(refstr[i_refstr].frac_coords)

    remappedstr = Structure(rescaledstr.lattice, splst, coordlst)

    return rmse, displst, remappedstr


def ChgDecoratorfromMag(structure, maglst, magdict, magrange):
    """
    The algorithm of assigning charge:

    1. Get all the possible charge state combinations from composition._get_oxid_state_guesses
    2. Find the most likely charge state according to magnetic moments and criteria of oxidation
       order.

    Args:
        Str: The structure that have
        maglst: The list of magnetic moment for each site
        magdict: The possible charge states for each element
        magrange: The typical range of magnetic moment

    Returns:
        decoratedstr
    """

    return decoratedstr
