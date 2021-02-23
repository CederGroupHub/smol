__author___ = 'Fengyu Xie'

"""
Mathematic utilities, including linear algebra, combinatorics and integerization
"""

import numpy as np
from itertools import combinations,product
from functools import reduce
from copy import deepcopy
import random
import math

from sympy.ntheory import factorint


def GCD(a,b):
    """ The Euclidean Algorithm, giving positive GCD's """
    if round(a)!=a or round(b)!=b:
        raise ValueError("GCD input must be integers!")
    a = abs(a)
    b = abs(b)
    while a:
        a, b = b%a, a
    return b    

def GCD_list(l):
    """ Find GCD of a list of numbers """
    if len(l)<1:
        return None
    elif len(l)==1:
        return l[0]
    else:
        return reduce(lambda a,b:GCD(a,b),l)

def LCM(a,b):
    if a==0 and b==0:
        return 0
    elif a==0 and b!=0:
        return b
    elif a!=0 and b==0:
        return a
    else:
        return a*b // GCD(a,b)

def LCM_list(l):
    if len(l)<1:
        return None
    elif len(l)==1:
        return l[0]
    else:
        return reduce(LCM,l)

# Combinatoric and intergerization
def reverse_ordering(l,ordering):
    """
    Given a mapping order of list, reverse that order
    and return the original list
    """
    original_l = [0 for i in range(len(l))]
    for cur_id,ori_id in enumerate(ordering):
        original_l[ori_id]=l[cur_id]
    return original_l

def rationalize_number(a,dim_limiter=100,dtol=1E-5):
    """
    Find a rational number near a within dtol.
    Returns the rational number in numerator/denominator
    form.
    Inputs:
        a: 
            float, a number to be rationalized
        dim_limiter: 
            maximum allowed denoninator. By default, 100
        dtol:
            maximum allowed difference of a to its rational
            form
    """
    if a==0:
        return 0,1
    for magnif in range(1,dim_limiter+1):
        a_prime = int(round(magnif*a))
        if abs(a_prime/magnif-a)<dtol:
            return a_prime,magnif
    raise ValueError("Can't find a rational number near {} within tolerance!".format(a))

def integerize_vector(v, dim_limiter=100,dtol=1E-5):
    """
    Ratioanlize all components of a vector v, and multiply the vector
    by lcm of all the component's denominator, so that all vector 
    components are converted to integers.
    We call this process 'intergerization' of a vector.
    Inputs:
        Same as rationalize_number.
    Outputs:
        v_int: 
            integerized vector. np.array(dtype=np.int64)
        magnification: 
            lcm of all the component's denominator. The magnification
            required to turn v into an integer vector.
    """
    denos = []
    v = np.array(v)
    for c in v:
        _,deno = rationalize_number(c,dim_limiter=dim_limiter,dtol=dtol)
        denos.append(deno)
    lcm = LCM_list(denos)
    return np.array(np.round(v*lcm),dtype=np.int64),lcm

def integerize_multiple(vs, dim_limiter=100,dtol=1E-5):
    """
    Integerize multiple vectors as a flattened vector.
    Return:
        np.array, int
    """
    vs = np.array(vs)
    shp = vs.shape
    vs_flatten = vs.flatten()
    vs_flat_int,mul = integerize_vector(vs_flatten,dim_limiter=dim_limiter,\
                                        dtol=dtol)
    vs_int = np.reshape(vs_flat_int,shp)
    return vs_int, mul

def combinatorial_number(n,m):
    """
    Calculate the combinatorial number when choosing m instances from a set of n instances.
    m,n: 
        integers.
    """
    if m>n:
        return 0
    return math.factorial(n)//(math.factorial(m)*math.factorial(n-m))

def get_integer_grid(subspc_normv,right_side=0,limiters=None):
    """
    Gives all integer grid points in a subspace (on a hyperplane defined by 
    a normal vector). The normal vector's components should all be positive,
    non-zero, and sorted (from low to high). This is called a standard form
    of the integer enumeration problem.
    Inputs:
        subspc_normv: positive integers, arraylike
        right_side: np.dot(subspc_normv,x) = right_side
        limiters: enumeration bounds of each dimension, list of tuples
                  [(lower_bound,upper_bound)]. Bounds will be taken.
                  All sections are closed sections.
    Outputs:
        A list of all integer grid points within limiter range, each in a list
        form.
    Note:
        This algorithm is far from optimal, so do not abuse it with overly high
        dimensionality!
    """
    d = len(subspc_normv)
    if limiters is None:
        limiters = [(-7,8) for i in range(d)]
    
    grids = []
    if d<1:
        raise ValueError('Dimensionality too low, can not enumerate!')

    elif d==1:
        k = subspc_normv[-1]
        if k!=0:
            if right_side%k ==0 and right_side//k>=limiters[-1][0] and\
               right_side//k<=limiters[-1][1]:
                grids.append([right_side//k])    
        else:
            if right_side == 0:
                for i in range(limiters[-1][0],limiters[-1][1]+1):
                    grids.append([i])

    else:
        new_limiters = limiters[:-1]
        #Move the last variable to the right hand side of hyperplane expression
        grids = []
        for val in range(limiters[-1][0],limiters[-1][1]+1):
            partial_grids = get_integer_grid(subspc_normv[:-1],right_side-val*subspc_normv[-1],\
                                             limiters=new_limiters)
            for p_grid in partial_grids:
                grids.append(p_grid+[val])

    #print('Grids:\n',grids,'\ndim:',d)
    return grids
      
def get_integer_basis(normal_vec,sl_flips_list=None):
    """
    The integer points on a hyperplane n x = 0 can form a lattice. This function can find
    the primitive lattice vectors with smallest norm, and are closest to orthogonal.
    Norm is defined by:
        L=sum_over_sublats(max_in_sublat(x_i))
    Inputs:
        normal_vec: normal vector of hyperplane. An arraylike of integers.
        sl_flips_list: define which components of x belong to flips on the same 
                     sublattice. A list of indices. If none, each flip will be 
                     considered to happend in an independent sublattice.
    OUtputs:
        Primitive lattice vectors. A list of np.int64 arrays
    """
    gcd = GCD_list(normal_vec)
    if gcd == 0:
        gcd = 1
    normal_vec = np.array(normal_vec,dtype=np.int64)//gcd
    #Make vector co-primal
    d = len(normal_vec)

    #Dimension of the Charge-neutral subspace
    if np.allclose(normal_vec,np.zeros(d)):
        D = d
    else:
        D = d-1

    if sl_flips_list is None:
        sl_flips_list = [[i] for i in range(d)]

    #Single out dimensions where normal vec components are 0. On these directions, basis
    #Is just a unit vector.
    zero_ids = np.where(normal_vec==0)[0]
    pos_ids = np.where(normal_vec < 0)[0]
    neg_ids = np.where(normal_vec > 0)[0]
    non_zero_ids = np.concatenate((pos_ids,neg_ids))

    unit_basis  = []
    for idx in zero_ids:
        e = np.zeros(d,dtype=np.int64)
        e[idx]=1
        unit_basis.append(e)

    d_remain = D-len(zero_ids)
    if d_remain == 0:
        return unit_basis
    else:
        #Convert the problem into standard form
        nv_partial = np.abs(normal_vec[non_zero_ids])
        table = list(enumerate(nv_partial))
        sorted_table = sorted(table,key=(lambda pair:pair[1]))
        nv_sorted = np.array([n for ori_id,n in sorted_table],dtype=np.int64)
        order_sorted = np.array([ori_id for ori_id,n in sorted_table],dtype=np.int64)
        #print('nv_sorted:',nv_sorted,'order_sorted:',order_sorted)

        #Estimate limiters
        d_nonzero = len(nv_partial)
        limiter_vectors = []
        for i,j in combinations(list(range(d_nonzero)),2):
            gcd = GCD(nv_sorted[i],nv_sorted[j])
            lcm = (nv_sorted[i]*nv_sorted[j])//gcd
            limiter_v = np.zeros(d_nonzero,dtype=np.int64)
            limiter_v[i] = lcm//nv_sorted[i]
            limiter_v[j] = -lcm//nv_sorted[j]
            limiter_vectors.append(limiter_v)
        #print('limiter_vectors:',limiter_vectors)

        limiter_ubs = np.max(np.vstack(limiter_vectors),axis=0)
        limiter_lbs = np.min(np.vstack(limiter_vectors),axis=0)
        limiters = list(zip(limiter_lbs,limiter_ubs))
        #print('limiters:',limiters)

        integer_grid = get_integer_grid(nv_sorted,limiters=limiters)
        
        basis_pool = []
        for point in integer_grid:
            reversely_ordered_point = reverse_ordering(point,order_sorted)
            basis = np.zeros(d,dtype=np.int64)
            for part_id,ori_id in enumerate(non_zero_ids):
                if ori_id in pos_ids:
                    basis[ori_id]=reversely_ordered_point[part_id]
                else:
                    basis[ori_id]=-1*reversely_ordered_point[part_id]
            basis_pool.append(basis)

        basis_pool = sorted(basis_pool,\
                            key=lambda v:\
                            (formula_norm(v,sl_flips_list=sl_flips_list),\
                             np.max(np.abs(v))))
        basis_pool = np.array(basis_pool)

        chosen_basis = []
        for v in basis_pool:
            #Full rank condition
            if np.linalg.matrix_rank(np.vstack(chosen_basis+[v]))==len(chosen_basis)+1:
                chosen_basis.append(v)
            if len(chosen_basis)==d_remain:
            #We have selected enough basis
                break

        return unit_basis+chosen_basis

def formula_norm(v,sl_flips_list=None):
    """L=sum_over_sublats(max_in_sublat(|x_i|))=total number of atoms flipped"""
    v = np.array(v)
    if sl_flips_list is None:
        sl_flips_list = [[i] for i in range(d)]  
    sl_form_sizes = []
    for sl in sl_flips_list:
        flip_nums = v[sl].tolist()+[-1*sum(v[sl])]
        sl_form_size = 0
        for num in flip_nums:
            if num>0:
                sl_form_size += num
        sl_form_sizes.append(sl_form_size)
    return sum(sl_form_sizes)

# Partition selection tools
def choose_section_from_partition(probs):
    """
    This function choose one section from a partition based on each section's
    normalized probability.
    Input:
        probs: 
            array-like, probabilities of each sections. If not normalized, will
            be normalized.
    Output:
        id: 
            The id of randomly chosen section.   
    """
    N_secs = len(probs)
    if N_secs<1:
        raise ValueError("Segment can't be selected!")

    norm_probs = np.array(probs)/np.sum(probs)
    upper_bnds = np.array([sum(norm_probs[:i+1]) for i in range(N_secs)])
    rand_seed = np.random.rand()

    for sec_id,sec_upper in enumerate(upper_bnds):
        if sec_id==0:
            sec_lower = 0
        else:
            sec_lower = upper_bnds[sec_id-1]
        if rand_seed>=sec_lower and rand_seed<sec_upper:
            return sec_id
    
    raise ValueError("Segment can't be selected.")

def enumerate_partitions(n_part,enum_fold,constrs=None,quota=1.0):
    """
    Recursivly enumerates possible partitions of a line section from 0.0 to 
    quota or from lower-bound to upper-bound if constrs is not None.
    Inputs:
        n_part(Int): 
            Number of partitions to be enumerated
        enum_fold(Int):
            Step of enumeration = quota/enum_fold.
        constrs(List of float tuples):
            lower and upper bound coustraints of each partition cut point.
            If None, just choose (0.0,quota) for each tuple.
        quota(float):
            Length of the line section to cut on.
    """
    if constrs is None:
        constrs = [(0.0,quota) for i in range(n_part)]

    lb,ub = constrs[0]
    ub = min(quota,ub)
    lb_int = int(np.ceil(lb*enum_fold))
    ub_int = int(np.floor(ub*enum_fold))

    if n_part < 1:
        raise ValueError("Can't partition less than 1 sections!")
    if n_part == 1:
        if quota == ub:
            return [[float(ub_int)/enum_fold]]
        else:
            return []

    this_level = [float(i)/enum_fold for i in range(lb_int,ub_int+1)]
    accumulated_enums = []
    for enum_x in this_level:
        next_levels = enumerate_partitions(n_part-1,enum_fold,\
                            constrs[1:],quota=quota-enum_x)
        if len(next_levels)!=0 and len(next_levels[0])==n_part-1:
            accumulated_enums.extend([[enum_x]+xs for xs in next_levels])

    return accumulated_enums
