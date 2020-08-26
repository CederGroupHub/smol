__author__ = "Fengyu Xie"

import numpy as np
import polytope as pc
from scipy.spatial import ConvexHull

from collections import OrderedDict
from itertools import combinations,product
from copy import deepcopy
from monty.json import MSONable, MontyDecoder
import json

from pymatgen import Composition

from smol.cofe.configspace.domain import Vacancy
from .utils import *

"""
This file contains functions related to implementing and navigating the 
compositional space.

In CEAuto, we first define a starting, charge-neutral, and fully occupied
('Vac' is also an occupation) composition in the compositional space.

Then we find all possible unitary, charge and number conserving flipping 
combinations of species, and define them as axis in the compositional space.

A composition is then defined as a vector, with each component corresponding
to the number of filps that have to be done on a 'flip axis' in order to get 
the target composition from the defined, starting composition.

For some supercell size, a composition might not be 'reachable' be cause 
supercell_size*atomic_ratioo is not an integer. For this case, you need to 
select a proper enumeration fold for your compositions(see enum_utils.py),
or choose a proper supercell size.

Case test in [[Li+,Mn3+,Ti4+],[P3-,O2-]] passed.
"""
NUMCONERROR = ValueError("Operation error, flipping not number conserved.") 
OUTOFSUBLATERROR = ValueError("Operation error, flipping between different sublattices.")
CHGBALANCEERROR = ValueError("Charge balance cannot be achieved with these species.")
OUTOFSUBSPACEERROR = ValueError("Given coordinate falls outside the subspace.")

SLACK_TOL = 1E-5

def decode_from_dict(d):
    return MontyDecoder().decode(json.dumps(d))


####
# Finding minimun charge-conserved, number-conserved flips to establish constrained
# coords system.
####

def get_unit_swps(bits):
    """
    Get all possible single site flips on each sublattice, and the charge changes that 
    they gives rise to.
    For example, 'Ca2+ -> Mg2+', and 'Li+ -> Mn2+'+'F- -> O2-' are all such type of 
    flips.
    Inputs:
        bits: 
            a list of Species or DummySpecies on each sublattice. For example:
            [[Specie.from_string('Ca2+'),Specie.from_string('Mg2+'))],
             [Specie.from_string('O2-')]]
    Outputs:
        unit_n_swps: 
            a flatten list of all possible, single site flips represented in n_bits, 
            each term written as:
            (flip_to_nbit,flip_from_nbit,sublattice_id_of_flip)
        chg_of_swps: 
            change of charge caused by each flip. A flatten list of integers
        swp_ids_in_sublat:
             a list specifying which sublattice should each flip belongs to. Will
             be used to set normalization constraints in comp space.
    """
    unit_swps = []
    unit_n_swps = []
    swp_ids_in_sublat = []
    cur_swp_id = 0
    for sl_id,sl_sps in enumerate(bits):
        unit_swps.extend([(sp,sl_sps[-1],sl_id) for sp in sl_sps[:-1]])
        unit_n_swps.extend([(sp_id,len(sl_sps)-1,sl_id) for sp_id in range(len(sl_sps)-1)])
        swp_ids_in_sublat.append([cur_swp_id+i for i in range(len(sl_sps)-1)])
        cur_swp_id += (len(sl_sps)-1)
        #(sp_before,sp_after,sublat_id)

    chg_of_swps = [int(p[0].oxi_state-p[1].oxi_state) for p in unit_swps]

    return unit_n_swps,chg_of_swps,swp_ids_in_sublat

def flipvec_to_operations(unit_n_swps, nbits, prim_lat_vecs):
    """
    This function translates flips from their vector from into their dictionary
    form.
    Each dictionary is written in the form below:
    {
     'from': 
           {sublattice_id:
               {specie_nbit_id: 
                   number_of_this_specie_to_be_anihilated_from_this_sublat
               }
                ...
           }
           ...
     'to':
           { 
           ...     numbr_of_this_specie_to_be_generated_on_this_sublat
           }
    }
    """
    n_sls = len(nbits)
    operations = []

    for flip_vec in prim_lat_vecs:
        operation = {'from':{},'to':{}}
        
        operation['from']={sl_id:{sp_id:0 for sp_id in nbits[sl_id]} for sl_id in range(n_sls)}
        operation['to'] = {sl_id:{sp_id:0 for sp_id in nbits[sl_id]} for sl_id in range(n_sls)}

        for flip,n_flip in zip(unit_n_swps,flip_vec):
            if n_flip > 0:
                flp_to,flp_from,sl_id = flip
                n = n_flip
            elif n_flip < 0:
                flp_from,flp_to,sl_id = flip
                n = -n_flip
            else:
                continue

            operation['from'][sl_id][flp_from] += n
            operation['to'][sl_id][flp_to] += n

        #Simplify ionic equations
        operation_clean = {'from':{},'to':{}}
        for sl_id in range(n_sls):
            for sp_id in nbits[sl_id]:
                del_n = operation['from'][sl_id][sp_id]-operation['to'][sl_id][sp_id]
                if del_n > 0:
                    if sl_id not in operation_clean['from']:
                        operation_clean['from'][sl_id]={}
                    operation_clean['from'][sl_id][sp_id]=del_n
                elif del_n < 0:
                    if sl_id not in operation_clean['to']:
                        operation_clean['to'][sl_id]={}
                    operation_clean['to'][sl_id][sp_id]= -del_n
                else:
                    continue

        operations.append(operation_clean)

    return operations    

def visualize_operations(operations,bits):
    """
    This function turns an operation dict into an string for easy visualization,
    """
    operation_strs = []
    for operation in operations:
        from_strs = []
        to_strs = []
        for sl_id in operation['from']:
            for swp_from,n in operation['from'][sl_id].items():
                from_name = str(bits[sl_id][swp_from])
                from_strs.append('{} {}({})'.format(n,from_name,sl_id))
        for sl_id in operation['to']:
            for swp_to,n in operation['to'][sl_id].items():
                to_name = str(bits[sl_id][swp_to])
                to_strs.append('{} {}({})'.format(n,to_name,sl_id))

        from_str = ' + '.join(from_strs)
        to_str = ' + '.join(to_strs)
        operation_strs.append(from_str+' -> '+to_str) 

    return operation_strs

####
# Compsitional space class
####

class CompSpace(MSONable):
    """
        This class generates a CN-compositional space from a list of Species or DummySpecies
        and sublattice sizes.

        A composition in CEAuto can be expressed in two forms:
        1, A Coordinate in unconstrained space, with 'single site flips' as basis vectors, and
           a 'background occupation' as the origin.
           We call this 'unconstr_coord'
        2, A Coordinate in constrained, charge neutral subspace, with 'charge neutral, number
           conserving elementary flips as basis vectors, and a charge neutral composition as 
           the origin.(Usually selected as one vertex of the constrained space.)
           We call this 'constr_coord'.

        For example, if bits = [[Li+,Mn3+,Ti4+],[P3-,O2-]] and sl_sizes = [1,1] (LMTOF rock-salt), then:
           'single site flips' basis are:
                Ti4+ -> Li+, Ti4+ -> Mn3+, O2- -> P3-
           'Background occupation' origin shall be:
                (Ti4+ | O-),supercell size =1
            The unconstrained space's dimensionality is 3.

           'charge neutral, number conserving elementary flips' bais shall be:
                3 Mn3+ -> 2 Ti4+ + Li+, Ti4+ + P3- -> O2- + Mn3+
           'charge neutral composition' origin can be chosen as:
                (Mn3+ | P-),supercell size = 1
            The constrained subspace's dimensionality is 2.

        Given composition:
            (Li0.5 Mn0.5| O), supercell size=1
            It's coordinates in the 1st system will be (0.5,0.5,0)
            In the second system that will be (0.5,1.0)

        When the system is always charge balanced (all the flips are charge conserved, background occu
        has 0 charge), then representation 1 and 2 are the same.

        Compspace class provides methods for you to convert between these two representations easily,
        write them into human readable form. It will also allow you to enumerate all possible integer 
        compositions given supercell size. Most importantly, it defines the CEAuto composition 
        enumeration method. For the exact way we do enumeration, please refer to the documentation of 
        each class methods.
            

    """
    def __init__(self,bits,sl_sizes=None):
        """
        Inputs:
            bits(List of Specie/DummySpecie): 
                bit list.
                Sorted before use. We don't sort it here in case the order 
                of Vacancy is broken, so be careful.
            sl_sizes: 
                Sublattice sizes in a PRIMITIVE cell. A list of integers. 
                len(bits)=# of sublats=len(sl_sizes).
                If None given, sl_sizes will be reset to [1,1,....]
        """
        self.bits = bits
        self.nbits = [list(range(len(sl_bits))) for sl_bits in bits]
        if sl_sizes is None:
            self.sl_sizes = [1 for i in range(len(self.bits))]
        elif len(sl_sizes)==len(bits):
            self.sl_sizes = sl_sizes
        else:
            raise ValueError("Sublattice number mismatch: check bits and sl_sizes parameters.")
  
        self.N_sts_prim = sum(self.sl_sizes)

        self.unit_n_swps,self.chg_of_swps,self.swp_ids_in_sublat = get_unit_swps(self.bits)

        self._constr_spc_basis = None
        self._constr_spc_vertices = None
        #Minimum supercell size required to make vetices coordinates all integer.
        self._polytope = None

        self._min_sc_size = None
        self._min_int_vertices = None
        self._min_grid = None
        self._int_grids = {}
    
    @property
    def bkgrnd_chg(self):
        """ 
        Type: Int

        Background charge. Defined of the charge in a supercell, when all sublattices
        are occupied by the last specie in its species list.
        """
        chg = 0
        for sl_bits,sl_size in zip(self.bits,self.sl_sizes):
            chg += int(sl_bits[-1].oxi_state)*sl_size
        return chg

    @property
    def unconstr_dim(self):
        """
        Type: int
       
        Dimensionality of the unconstrained space
        """
        return len(self.unit_n_swps)
 
    @property
    def is_charge_constred(self):
        """
        Type: Boolean
 
        If true, charge neutrality is rank-1, and reduces allowed space dim by 1.
        Otherwise charge neutrality is 0==0, and does not affect the space.
        """
        d = len(self.chg_of_swps)
        return not(np.allclose(np.zeros(d),self.chg_of_swps) and self.bkgrnd_chg==0)

    @property
    def dim(self):
        """
        Type: Boolean

        Dimensionality of the allowed conpositional space.
        """
        d = self.unconstr_dim
        if not self.is_charge_constred:
            return d
        else:
            return d-1

    @property
    def constr_spc_basis(self):
        """
        Get 'minimal charge-neutral flips basis' in vector representation. 
        Given any compositional space, all valid, charge-neutral compoisitons are 
        integer grids on this space or its subspace. What we do is to get the primitive
        lattice vectors of the lattice defined by these grid points.
        For example:
        [[Li+,Mn3+,Ti4+],[P3-,O2-]] system, minimal charge and number conserving flips 
        are:
        3 Mn3+ <-> Li+ + 2 Ti4+, 
        Ti4+ + P3- <-> Mn3+ + O2- 
        Their vector forms are:
        (1,-3,0), (0,1,-1)  

        Type: 2d np.array of np.int64
        """        
        if self._constr_spc_basis is None:
            self._constr_spc_basis = \
                 np.array(get_integer_basis(self.chg_of_swps,sl_flips_list=self.swp_ids_in_sublat),dtype=np.int64)
        return self._constr_spc_basis

    @property
    def min_flips(self):
        """
        Dictionary representation of minimal charge conserving flips.
        """
        _operations = flipvec_to_operations(self.unit_n_swps,\
                                            self.nbits,\
                                            self.constr_spc_basis)
        return _operations

    @property
    def min_flip_strings(self):
        """
        Human readable minial charge conserving flips, written in ionic equations.
        """
        return visualize_operations(self.min_flips,self.bits)

    @property
    def polytope(self):
        """
        Express the configurational space (supercellsize=1) as a polytope.Polytope object.
        Shall be expressed in type 2 basis

        R and t are rotation matrix and translation vector to transform constrained to unconstrained
        basis.
        To transform a constrained basis to unconstrained basis, use:
            x = R.T @ x'.append(0) + t
        """
        if self._polytope is None:
            facets_unconstred = []
            for sl_flp_ids,sl_size in zip(self.swp_ids_in_sublat,self.sl_sizes):
                a = np.zeros(self.unconstr_dim)
                a[sl_flp_ids]=1
                bi = sl_size
                facets_unconstred.append((a,bi))
            #sum(x_i) for i in sublattice <= 1
            A_n = np.vstack([a for a,bi in facets_unconstred])
            b_n = np.array([bi for a,bi in facets_unconstred])
            # x_i >=0 for all i
            A = np.vstack((A_n,-1*np.identity(self.unconstr_dim)))
            b = np.concatenate((b_n,np.zeros(self.unconstr_dim)))
 
            if not self.is_charge_constred:
                #polytope = pc.Polytope(A,b) Ax<=b.
                R = np.idendity(self.unconstr_dim)
                t = np.zeros(self.unconstr_dim)
                self._polytope = (A,b,R,t)          
            else:
                # x-t = R.T * x', where x'[-1]=0. Dimension reduced by 1.
                # We have to reduce dimension first because polytope package
                # can not handle polytope in a subspace. It will consider the
                # subspace as an empty set.

                # x: unconstrained, x': constrained
                R = np.vstack((self.constr_spc_basis,np.array(self.chg_of_swps)))
                t = np.zeros(self.unconstr_dim)
                t[0] = -self.bkgrnd_chg/self.chg_of_swps[0]
                A_sub = A@R.T
                A_sub = A_sub[:,:-1]
                #slice A, remove last col, because the last component of x' will
                #always be 0
                b_sub = b-A@t
                self._polytope = (A_sub,b_sub,R,t)
        return self._polytope

    """ A, b , R, t all np.arrays"""
    @property
    def A(self):
        return self.polytope[0]

    @property
    def b(self):
        return self.polytope[1]
   
    #R and t are only valid in unit comp space (sc_size=1)!!!
    @property
    def R(self):
        return self.polytope[2]

    @property
    def t(self):
        return self.polytope[3]

    def _is_in_subspace(self,x,sc_size=1):
        """
        Given an unconstrained coordinate and its corresponding supercell size,
        check if it is in the constraint subspace.
        Returns a boolean.

        SLACK_TOL:
            Maximum allowed slack to constraints
        """
        x_scaled = np.array(x)/sc_size
        try:
            x_prime = self._unconstr_to_constr_coords(x_scaled,sc_size=1)
            return True
        except:
            return False

    def constr_spc_vertices(self,form='unconstr'):
        """
        Find extremums of the constrained compositional space in a primitive cell,

        Type: 2D np.array of floats

        Inputs:
            form (string):
                Specifies the format to output the compositions.
                'unconstr': use unconstrained (type 1) coordinates.(default)
                'constr': use constrained (type 2) coordinates.
                'compstat': use compstat lists.(See self._unconstr_to_compstat doc)
                'composition': use a pymatgen.composition for each sublattice 
                               (vacancies not explicitly included)
        """
        if self._constr_spc_vertices is None:
            if not self.is_charge_constred:
                A,b,_,_=self.polytope
                poly = pc.Polytope(A,b)
                self._constr_spc_vertices = pc.extreme(poly)
            else:
                A,b,R,t=self.polytope
                poly_sub = pc.Polytope(A,b)
                vert_sub = pc.extreme(poly_sub)
                n = vert_sub.shape[0]
                vert = np.hstack((vert_sub,np.zeros((n,1))))
                #Transform back into unconstraned coord
                self._constr_spc_vertices = vert@R + t

        if len(self._constr_spc_vertices)==0:
            raise CHGBALANCEERROR

        #This function formuates multiple unconstrained coords together.
        return self._formulate_unconstr(self._constr_spc_vertices,form=form,sc_size=1)

    @property
    def min_sc_size(self):
        """
        Minimal supercell size to get integer composition.
        In this function we also vertices of the compositional space after all coordinates
        are multiplied by self.min_sc_size. 
        """
        if self._min_sc_size or self._min_int_vertices is None:
            self._min_int_vertices, self._min_sc_size = \
                integerize_multiple(self.constr_spc_vertices())
        return self._min_sc_size

    def min_int_vertices(self,form='unconstr'):
        """
        Type: 2D np.array of np.int64
        minimal integerized compositional space vertices (unconstrained format).
        Inputs:
            form (string):
                Specifies the format to output the compositions.
                'unconstr': use unconstrained (type 1) coordinates.(default)
                'constr': use constrained (type 2) coordinates.
                'compstat': use compstat lists.(See self._unconstr_to_compstat doc)
                'composition': use a pymatgen.composition for each sublattice 
                               (vacancies not explicitly included)
        """
        if self._min_sc_size or self._min_int_vertices is None:
            min_sc_size = self.min_sc_size

        return self._formulate_unconstr(self._min_int_vertices,form=form,sc_size=self.min_sc_size)

    def int_vertices(self,sc_size=1,step=1,form='unconstr'):
        """
        Type: np.array of np.int64
        If supercell size is a multiple of min_sc_size, then int_vertices are just min_int_vertices*multiple;
        Otherwise int_vertices are taken as convex hull vertices of set self.int_grids(sc_size)
        Inputs:
            sc_size(int): supercell sizes to enumerate integer composition on
            form (string):
                Specifies the format to output the compositions.
                'unconstr': use unconstrained (type 1) coordinates.(default)
                'constr': use constrained (type 2) coordinates.
                'compstat': use compstat lists.(See self._unconstr_to_compstat doc)
                'composition': use a pymatgen.composition for each sublattice 
                               (vacancies not explicitly included)
        """
        if sc_size%self.min_sc_size == 0:
            vertices = self.min_int_vertices()*(sc_size//self.min_sc_size)
        else:
            int_grids = np.array(self.int_grids(sc_size=sc_size),dtype=np.int64)
            grids_in_constr_spc = np.array([self._unconstr_to_constr_coords(x,sc_size=sc_size) for x in int_grids])
            try:
                hull = ConvexHull(grids_in_constr_spc)
                vertices = int_grids[hull.vertices]
                vertices = np.array(vertices,dtype=np.int64)
            except:
                #points does not satisfy the requirement to form a hull in the constrained space
                vertices = int_grids

        return self._formulate_unconstr(vertices,form=form,sc_size=sc_size)

    def min_grid(self,form='unconstr'):
        """
        Get the minimum compositional grid: multiply the primitive cell compositional space
        by self.min_sc_size, and find all the integer grids in the new, supercell.

        Type: 2D np.array of np.int64
        Inputs:
            form (string):
                Specifies the format to output the compositions.
                'unconstr': use unconstrained (type 1) coordinates.(default)
                'constr': use constrained (type 2) coordinates.
                'compstat': use compstat lists.(See self._unconstr_to_compstat doc)
                'composition': use a pymatgen.composition for each sublattice 
                               (vacancies not explicitly included)
        """
        if self._min_grid is None:
            self._min_grid = self._enum_int_grids(sc_size=self.min_sc_size)

        return self._formulate_unconstr(self._min_grid,form=form,sc_size=self.min_sc_size)

    def int_grids(self,sc_size=1,form='unconstr'):
        """
        Get the integer grid in a supercell's compositional space. These are allowed integer
        compositions.
   
        Type: 2D np.array of np.int64
        Inputs:
            sc_size (int):
                Supercell size to enumerate on.
            form (string):
                Specifies the format to output the compositions.
                'unconstr': use unconstrained (type 1) coordinates.(default)
                'constr': use constrained (type 2) coordinates.
                'compstat': use compstat lists.(See self._unconstr_to_compstat doc)
                'composition': use a pymatgen.composition for each sublattice 
                               (vacancies not explicitly included)

        Note: if you want to stepped enumeration, just divide sc_size by step, and multiply
              the resulted array with step. (You don't even need to multiply back, when
              formula = 'composition', because this automatically gives you fractional 
              composition)
        """
        if sc_size not in self._int_grids:
            self._int_grids[sc_size] = self._enum_int_grids(sc_size=sc_size)

        return self._formulate_unconstr(self._int_grids[sc_size],form=form,sc_size=sc_size)

    def _enum_int_grids(self,sc_size=1):
        """
        Returns: list of lists of ints
        Enumerate all possible compositions in charge-neutral space.
        Input:
            sc_size: the supercell size to enumerate integer compositions on.
                     Recommended to be a multiply of self.min_sc_size, otherwise
                     we can't guarantee to find any integer composition.
        """
        if sc_size % self.min_sc_size == 0:
            magnif = sc_size//self.min_sc_size
            int_vertices = self.min_int_vertices()*magnif
            limiters_ub = np.max(int_vertices,axis=0)
            limiters_lb = np.min(int_vertices,axis=0)

        else:
            #Then integer composition is not guaranteed to be found.
            vertices = self.constr_spc_vertices()*sc_size
            limiters_ub = np.array(np.ceil(np.max(vertices,axis=0)),dtype=np.int64)
            limiters_lb = np.array(np.floor(np.min(vertices,axis=0)),dtype=np.int64)

        limiters = list(zip(limiters_lb,limiters_ub))
        right_side = -1*self.bkgrnd_chg*sc_size
        grid = get_integer_grid(self.chg_of_swps,right_side=right_side,\
                                limiters = limiters)

        #print(grid)
        enum_grid = []
        for p in grid:
            if self._is_in_subspace(p,sc_size=sc_size):
                enum_grid.append(p)

        return np.array(enum_grid,dtype=np.int64)

    def frac_grids(self,sc_size=1,form='unconstr'):
        """
        Enumerate integer compositions under a certain sc_size, and turn it into
        float form by normalizeing with sc_size.
        Inputs:
            sc_size (int):
                Supercell size to numerate on.
            form (string):
                Specifies the format to output the compositions.
                'unconstr': use unconstrained (type 1) coordinates.(default)
                'constr': use constrained (type 2) coordinates.
                'compstat': use compstat lists.(See self._unconstr_to_compstat doc)
                'composition': use a pymatgen.composition for each sublattice 
                               (vacancies not explicitly included)
        Note: if you want to stepped enumeration, just divide sc_size by step, and multiply
              the resulted array with step. (You don't even need to multiply back, when
              formula = 'composition', because this automatically gives you fractional 
              composition)
        """
        comps = np.array(self.int_grids(sc_size),dtype=np.float64)/sc_size

        return self._formulate_unconstr(comps,form=form,sc_size=1)

    def _unconstr_to_constr_coords(self,x,sc_size=1):
        """
        Unconstrained coordinate system to constrained coordinate system.
        In constrained coordinate system, a composition will be written as
        number of flips required to reach this composition from a starting 
        composition.        

        This one does not check if one specific point is in subspace

        to_int: if true, round coords to integers.

        Outputs:
            x_prime: constrained coordinates vector, in its proper dimension
            d_slack: slack distance out of constraint plane. Will always be 0 if charge 
                   constraint does not apply.
                   if d_slack>SLACK_TOL, then we consider the charge constraint as broken.
        """
        #scale down to unit comp space
        x = np.array(x)/sc_size
        if not self.is_charge_constred:
            x_prime = deepcopy(x)
            d_slack = 0
        else:
            x_prime = np.linalg.inv((self.R).T)@(x-self.t)
            d_slack = x_prime[-1]
            x_prime = x_prime[:-1]
            
        b = self.A@x_prime
        for bi_p,bi in zip(b,self.b):
            if bi_p-bi > SLACK_TOL:
                raise OUTOFSUBSPACEERROR

        if abs(d_slack) > SLACK_TOL:
            raise OUTOFSUBSPACEERROR

        #scale back up to sc_size
        x_prime = x_prime*sc_size
        d_slack = d_slack*sc_size

        return x_prime

    def _constr_to_unconstr_coords(self,x_prime,sc_size=1,to_int=False):
        """
        Constrained coordinate system to unconstrained coordinate system.
        """
        #scale down to unit comp space
        x_prime = np.array(x_prime)/sc_size

        b = self.A@x_prime
        for bi_p,bi in zip(b,self.b):
            if bi_p-bi > SLACK_TOL:
                raise OUTOFSUBSPACEERROR

        if not self.is_charge_constred:
            x = deepcopy(x_prime)
        else:
            x = deepcopy(x_prime)
            x = np.concatenate((x,np.array([0])))
            x = (self.R).T@x + self.t
       
        #scale back up
        x = x*sc_size

        if to_int:
            x = np.round(x)
            x = np.array(x_prime,dtype=np.int64)

        return x
 
    def _unconstr_to_compstat(self,x,sc_size=1):
        """
        Translate unconstrained coordinate to statistics of specie numbers on 
        each sublattice. Will have the same shape as self.nbitsa

        Return:
            Compstat: List of lists of int.
        """
        v_id = 0
        compstat = [[0 for i in range(len(sl_nbits))] for sl_nbits in self.nbits]
        for sl_id,sl_nbits in enumerate(self.nbits):
            sl_sum = 0
            for b_id,bit in enumerate(sl_nbits[:-1]):
                compstat[sl_id][b_id] = x[v_id]
                sl_sum += x[v_id]
                v_id +=1
            compstat[sl_id][-1] = self.sl_sizes[sl_id]*sc_size - sl_sum
            if compstat[sl_id][-1] < 0:
                raise OUTOFSUBSPACEERROR

        return compstat

    def _unconstr_to_composition(self,x,sc_size=1):
        """
        Translate an unconstranied coordinate into a list of composition dictionaries 
        by each sublattice. Vacancies are not explicitly included for the convenience 
        of structure generator.
        """
        compstat = self._unconstr_to_compstat(x,sc_size=sc_size)
        sl_sizes = np.array(self.sl_sizes)*sc_size

        sl_comps = []
        for sl_id,sl_bits in enumerate(self.bits):
            sl_comp = {}

            for b_id, bit in enumerate(sl_bits):
                #Trim vacancies from the composition, for pymatgen to read the structure.
                if isinstance(bit, Vacancy):
                    continue
                sl_comp[bit] = float(compstat[sl_id][b_id])/sl_sizes[sl_id]

            sl_comps.append(Composition(sl_comp))

        return sl_comps

    def _formulate_unconstr(self,arr,form='unconstr',sc_size=1):
        """
        Translates an unconstrained coordinate into different forms.
        Inputs:
            arr: must be 2D np.array
            sc_size (int):
                Supercell size to numerate on.
            form (string):
                Specifies the format to output the compositions.
                'unconstr': use unconstrained (type 1) coordinates.(default)
                'constr': use constrained (type 2) coordinates.
                'compstat': use compstat lists.(See self._unconstr_to_compstat doc)
                'composition': use a pymatgen.composition for each sublattice 
                               (vacancies not explicitly included)
        """
        if form == 'unconstr':
            #Output is 2D array
            return arr
        elif form == 'constr':
            #Output is 2D array
            return np.array([self._unconstr_to_constr_coords(x,sc_size=sc_size)
                             for x in arr])
        elif form == 'compstat':
            #Output is a list of 2D lists, each 2D list is for an element in the
            #Original 2D array
            return [self._unconstr_to_compstat(x,sc_size=sc_size) for x in arr]
        elif form == 'composition':
            #Output will be a 2D list of pymatgen.Composition
            return [self._unconstr_to_composition(x,sc_size=sc_size) for x in arr]
        else:
            raise ValueError('Requested format not supported.')


    def as_dict(self):
        bits_d = [[sp.as_dict() for sp in sl_sps] for sl_sps in self.bits]
        # polytope is a tuple of np.arrays        
        poly = [item.tolist() for item in self.polytope]
        int_grids = {key:val.tolist() for key,val in self._int_grids.items()}

        return {
                'bits': bits_d,
                'sl_sizes': self.sl_sizes,
                'constr_spc_basis': self.constr_spc_basis.tolist(),
                'constr_spc_vertices': self.constr_spc_vertices().tolist(),
                'polytope': poly,
                'min_sc_size': self.min_sc_size,
                'min_int_vertices': self.min_int_vertices().tolist(),
                'min_grid': self.min_grid().tolist(),
                'int_grids': int_grids,
                '@module': self.__class__.__module__,
                '@class': self.__class__.__name__
               }

    @classmethod
    def from_dict(cls,d):
        bits = [[decode_from_dict(sp_d) for sp_d in sl_sps] for sl_sps in d['bits']]
        
        obj = cls(bits,d['sl_sizes'])        
 
        if 'constr_spc_basis' in d:
            obj._constr_spc_basis = np.array(d['constr_spc_basis'],dtype=np.int64)
            

        if 'constr_spc_vertices' in d:          
            obj._constr_spc_vertices = np.array(d['constr_spc_vertices'])

        if 'polytope' in d:            
            poly = d['polytope']
            poly = [np.array(item) for item in poly]
            obj._polytope = poly

        if 'min_sc_size' in d:
            obj._min_sc_size = d['min_sc_size']

        if 'min_int_vertices' in d:
            obj._min_int_vertices = np.array(d['min_int_vertices'],dtype=np.int64)

        if 'min_grid' in d:
            obj._min_grid = np.array(d['min_grid'],dtype=np.int64)
 
        if 'int_grids' in d:
            int_grids = d['int_grids']
            obj._int_grids = {key:np.array(val,dtype=np.int64) for key,val in int_grids}

        return obj
