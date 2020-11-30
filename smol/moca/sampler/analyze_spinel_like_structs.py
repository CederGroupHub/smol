import json
import numpy as np

with open('dr_spinel.json', 'r') as fin:
    d = json.load(fin)

def analyze_structs(data):
    m = np.matrix([[2,2,-2],[-2,2,2],[2,-2,2]])
    # default transformation scs is 3x3x3
    transformation = np.matrix([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
    scs = m @ transformation
    size = np.linalg.det(scs)
    for T in data:
        for step in data[T]:
            features_normalized = step / int(size)
