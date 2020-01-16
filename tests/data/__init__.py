import os
import json
from pymatgen import Structure

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

with open(os.path.join(DATA_DIR, 'lno_prim.json'), 'r') as f:
    lno_prim = Structure.from_dict(json.load(f))

with open(os.path.join(DATA_DIR, 'lno_fitting_data.json'), 'r') as f:
    lno_data = [(Structure.from_dict(x['s']), x['toten']) for x in json.load(f)]
