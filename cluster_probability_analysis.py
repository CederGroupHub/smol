import numpy as np
import json

__author__ = "Julia Yang"
__date__ = "2020/11/06"

"""
For Zinab's Li-V-O system.   

Given two lattice sites (binary disorder and quaternary disorder), 
calculate the following cluster probabilities given the generator function 
and V-matrix solutions: 

1. Tetrahedral Li 
2. Octahedral Li/V4+/V5+ 
3. Face-sharing Li-Li 
4. Face-sharing Li-V4+
5. Face-sharing Li-V5+
6. 0-TM quadruplets
7. 1-TM quadruplets
8. 2-TM quadruplets

"""
#################################################
############V-MATRIX SOLUTIONS###################

tetM = 2
li_tet = np.array([-1]) * tetM
vac_tet = np.array([1])

octM = 4
li_oct = np.array([-0.5, 0, -0.25]) * octM
v4_oct = np.array([0, -0.5, 0.25]) * octM
v5_oct = np.array([0.5, 0, -0.25]) * octM
vac_oct = np.array([0, 0.5, 0.25]) * octM

#################################################
######### critical fnc and information ##########

with open('20201105_ZJ_group_ids.json', 'r') as fin:
    group_ids = json.load(fin)

def calculate_subcluster(point_terms, group_number, corr):
    decorations = group_ids[group_number]
   # now take the fancy dot product
    total = 0
    for counter in decorations:
        for bcOrdering in decorations[counter]:
            i = -1
            v_i = 1.0
            for bc in bcOrdering:
                i += 1
                v_i *= point_terms[i][bc] # order of point_terms matters
            total += (v_i * corr[int(counter)]) # per basis function
    return total

#################################################
# calculate single-site concentrations #
startTet, endTet = None, None # user inputs this (start=first tet basis function, end = last tet basis function +1)
startOct, endOct = None, None # user inputs this (start=first oct basis function, end = last oct basis function +1)

def calculate_LiTet(corr):
    return 1 / tetM * (1 + np.dot(li_tet, corr[startTet:endTet]))

def calculate_LiOct(corr):
    return 1 / octM * (1 + np.dot(li_oct, corr[startOct:endOct]))

def calculate_V4Oct(corr):
    return 1 / octM * (1 + np.dot(v4_oct, corr[startOct:endOct]))

def calculate_V5Oct(corr):
    return 1 / octM * (1 + np.dot(v5_oct, corr[startOct:endOct]))

def calculate_VacOct(corr):
    return 1 / octM * (1 + np.dot(vac_oct, corr[startOct:endOct]))

#################################################
# calculate face-sharing concentrations #
startTet, endTet = None, None # user inputs this (start=first tet basis function, end = last tet basis function +1)
startOct, endOct = None, None # user inputs this (start=first oct basis function, end = last oct basis function +1)
fs_groupID = None # user inputs this

def calculate_LiLi_fs(corr, fs_groupID):
    return (1 / tetM * 1 / octM * (1 + np.dot(li_tet, corr[startTet:endTet]) +
                                       np.dot(li_oct, corr[startOct:endOct]) +
                                       calculate_subcluster([li_tet, li_oct],
                                                  str(fs_groupID), corr)))

def calculate_LiV4_fs(corr, fs_groupID):
    return (1 / tetM * 1 / octM * (1 + np.dot(li_tet, corr[startTet:endTet]) +
                                       np.dot(v4_oct, corr[startOct:endOct]) +
                                       calculate_subcluster([li_tet, v4_oct],
                                                  str(fs_groupID), corr)))

def calculate_LiV5_fs(corr, fs_groupID):
    return (1 / tetM * 1 / octM * (1 + np.dot(li_tet, corr[startTet:endTet]) +
                                       np.dot(v5_oct, corr[startOct:endOct]) +
                                       calculate_subcluster([li_tet, v5_oct],
                                                  str(fs_groupID), corr)))

#################################################
# calculate 0/1/2TM concentrations #
oct_oct_groupID = None # user inputs this
oct_oct_oct_groupID = None # user inputs this
oct_oct_oct_oct_groupID = None # user inputs this

startOct, endOct = None, None # user inputs this (start=first oct basis function, end = last oct basis function +1)

def calculate_XTM(corr):
    # 4-TM channel function. Since you only need the 0, 1, 2-TM channels only those are returned
    keys = {0: 1, 1: 0, 2: 4, 3: 5}
    lowestTMCharge = 4
    l = -1
    TM1 = 0  # (1, 1, 1, 2)
    TM3 = 0  # (1, 2, 2, 2)
    TM2 = 0  # (1, 1, 2, 2)
    TM0 = 0  # (1, 1, 1, 1)
    TM4 = 0  # (2, 2, 2, 2)

    for oct_1 in [li_oct, vac_oct, v4_oct, v5_oct]:
        l += 1
        i = -1
        for oct_2 in [li_oct, vac_oct, v4_oct, v5_oct]:
            i += 1
            j = -1
            for oct_3 in [li_oct, vac_oct, v4_oct, v5_oct]:
                j += 1
                k = -1
                for oct_4 in [li_oct, vac_oct, v4_oct, v5_oct]:
                    k += 1
                    current = tuple(sorted((keys[i], keys[j], keys[k], keys[l])))
                    tot = 1 / octM ** 4 * (1 + np.dot(oct_1, corr[startOct:endOct]) +
                                        np.dot(oct_2, corr[startOct:endOct]) +
                                        np.dot(oct_3, corr[startOct:endOct]) +
                                        np.dot(oct_4, corr[startOct:endOct]) +

                                        calculate_subcluster([oct_1, oct_2],
                                                             str(oct_oct_groupID), corr) +
                                        calculate_subcluster([oct_1, oct_3],
                                                             str(oct_oct_groupID), corr) +
                                        calculate_subcluster([oct_1, oct_4],
                                                             str(oct_oct_groupID), corr) +
                                        calculate_subcluster([oct_2, oct_3],
                                                             str(oct_oct_groupID), corr) +
                                        calculate_subcluster([oct_2, oct_4],
                                                             str(oct_oct_groupID), corr) +
                                        calculate_subcluster([oct_3, oct_4],
                                                             str(oct_oct_groupID), corr) +

                                        calculate_subcluster([oct_1, oct_2, oct_3],
                                                             str(oct_oct_oct_groupID), corr) +
                                        calculate_subcluster([oct_4, oct_1, oct_2],
                                                             str(oct_oct_oct_groupID), corr) +
                                        calculate_subcluster([oct_2, oct_3, oct_4],
                                                             str(oct_oct_oct_groupID), corr) +
                                        calculate_subcluster([oct_3, oct_4, oct_1],
                                                             str(oct_oct_oct_groupID), corr) +

                                        calculate_subcluster([oct_1, oct_2, oct_3,
                                                              oct_4],
                                                             str(oct_oct_oct_oct_groupID), corr)
                                        )
                    if current[0] >= lowestTMCharge:
                        TM4 += tot
                    else:
                        if current[1] >= lowestTMCharge:
                            TM3 += tot
                        else:
                            if current[2] >= lowestTMCharge:
                                TM2 += tot
                            else:
                                if current[3] >= lowestTMCharge:
                                    TM1 += tot
                                else:
                                    TM0 += tot
    return (TM0, TM1, TM2)

def calculate_2TM_wLiVac(corr):
    # 4-TM channel function. Since you only need the 0, 1, 2-TM channels only those are returned
    keys = {0: 1, 1: 0, 2: 4, 3: 5}
    lowestTMCharge = 4
    l = -1
    TM1 = {'1Li3vac': 0, '2Li2Vac': 0, '3Li1Vac': 0}  # (1, 1, 1, 2)
    TM3 = 0  # (1, 2, 2, 2)
    TM2 = {'1Li1Vac': 0, '2Li0Vac': 0}   # (1, 1, 2, 2)
    TM0 = {'1Li3Vac': 0, '2Li2Vac': 0, '3Li1Vac': 0, '4Li0Vac': 0}  # (1, 1, 1, 1)
    TM4 = 0  # (2, 2, 2, 2)

    for oct_1 in [li_oct, vac_oct, v4_oct, v5_oct]:
        l += 1
        i = -1
        for oct_2 in [li_oct, vac_oct, v4_oct, v5_oct]:
            i += 1
            j = -1
            for oct_3 in [li_oct, vac_oct, v4_oct, v5_oct]:
                j += 1
                k = -1
                for oct_4 in [li_oct, vac_oct, v4_oct, v5_oct]:
                    k += 1
                    current = tuple(sorted((keys[i], keys[j], keys[k], keys[l])))
                    tot = 1 / octM ** 4 * (1 + np.dot(oct_1, corr[startOct:endOct]) +
                                        np.dot(oct_2, corr[startOct:endOct]) +
                                        np.dot(oct_3, corr[startOct:endOct]) +
                                        np.dot(oct_4, corr[startOct:endOct]) +

                                        calculate_subcluster([oct_1, oct_2],
                                                             str(oct_oct_groupID), corr) +
                                        calculate_subcluster([oct_1, oct_3],
                                                             str(oct_oct_groupID), corr) +
                                        calculate_subcluster([oct_1, oct_4],
                                                             str(oct_oct_groupID), corr) +
                                        calculate_subcluster([oct_2, oct_3],
                                                             str(oct_oct_groupID), corr) +
                                        calculate_subcluster([oct_2, oct_4],
                                                             str(oct_oct_groupID), corr) +
                                        calculate_subcluster([oct_3, oct_4],
                                                             str(oct_oct_groupID), corr) +

                                        calculate_subcluster([oct_1, oct_2, oct_3],
                                                             str(oct_oct_oct_groupID), corr) +
                                        calculate_subcluster([oct_4, oct_1, oct_2],
                                                             str(oct_oct_oct_groupID), corr) +
                                        calculate_subcluster([oct_2, oct_3, oct_4],
                                                             str(oct_oct_oct_groupID), corr) +
                                        calculate_subcluster([oct_3, oct_4, oct_1],
                                                             str(oct_oct_oct_groupID), corr) +

                                        calculate_subcluster([oct_1, oct_2, oct_3,
                                                              oct_4],
                                                             str(oct_oct_oct_oct_groupID), corr)
                                        )
                    if current[0] >= lowestTMCharge:
                        TM4 += tot
                    else:
                        if current[1] >= lowestTMCharge:
                            TM3 += tot
                        else:
                            if current[2] >= lowestTMCharge:
                                if current.count(1) == 2: TM2['2Li0Vac'] += tot
                                elif current.count(1) == 1 and current.count(0) == 1: TM1['1Li1Vac'] += tot
                            else:
                                if current[3] >= lowestTMCharge:
                                    if current.count(1) == 3: TM1['3Li0Vac'] += tot
                                    elif current.count(1) == 2 and current.count(0) == 1: TM1['2Li1Vac'] += tot
                                    elif current.count(1) == 1 and current.count(0) == 2: TM1['1Li2Vac'] += tot
                                else:
                                    if current.count(1) == 4: TM0['4Li0Vac'] += tot
                                    elif current.count(1) == 3: TM0['3Li1Vac'] += tot
                                    elif current.count(1) == 2: TM0['2Li2Vac'] += tot
                                    elif current.count(1) == 1: TM0['1Li3Vac'] += tot
    return (TM0, TM1, TM2)