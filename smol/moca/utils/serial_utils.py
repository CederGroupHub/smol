"""
Serialization helper functions
"""

__author__ = 'Fengyu Xie'

from monty.json import MontyDecoder
import json
import numpy as np

from pymatgen import Composition

# Monty decoding any dict
def decode_from_dict(d):
    return MontyDecoder().decode(json.dumps(d))

#serialize and de-serialize compositions
def serialize_comp(comp):
    """
    Serialize a pymatgen.Composition or a list of pymatgen.Composition.
    Composition.as_dict only keeps species string. This will keep all
    property informations of a specie.
    Inputs:
        comp(Composition or List of Composition):
            the composition to serialize
    Outputs:
        A serialzed composition, in form: [(specie_dict,num_specie)] or 
        a list of such.
    """
    if isinstance(comp,list):
        return [serialize_comp(sl_comp) for sl_comp in comp]

    return [(sp.as_dict(),n) for sp,n in comp.items()]

def deser_comp(comp_ser):
    """
    Deserialize to a pymatgen.Composition or a list of pymatgen.Composition.
    Composition.as_dict only keeps species string. This will keep all
    property informations of a specie.
    Inputs:
        comp_ser(List of tuples or List of List of tuples):
        A serialzed composition, in form: [(specie_dict,num_specie)] or 
        a list of such.
        
    Outputs:
        comp(Composition or List of Composition):
            the composition to serialize
    """
    if isinstance(comp_ser,list) and isinstance(comp_ser[0],list):
        return [deser_comp(sl_comp_ser) for sl_comp_ser in comp_ser]

    return Composition({decode_from_dict(sp_d):n for sp_d,n in comp_ser})

#Serialize anything
def serialize_any(obj):
    """
    Serialize any object or list of objects.
    Args:
        obj: 
            An object, or a multi-dimensional list of objects.
    Returns:
        If obj is a multi-dimensional list, will have the same shape
        as input.
        If obj is a single object, returns a single dicitonary.
    Do not use this for composition!
    """
    if isinstance(obj,(list,tuple,set,np.ndarray)):
        return serialize_any([o for o in obj])

    if 'as_dict' in dir(obj):
        return obj.as_dict()

    return obj #Not serializable, or serialization not needed.
