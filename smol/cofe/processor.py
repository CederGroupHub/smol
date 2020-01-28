"""
Implementation of a processor for a fixed size Super Cell optimized to compute
correlation vectors and local changes in correlation vectors. This class allows
the use a cluster expansion hamiltonian to run Monte Carlo based simulations.
The name comes from its use to create the "Markov process" aka Markov Processor
"""

from monty.json import MSONable


class Processor(MSONable):
    """
    A processor used to generate Markov processes for sampling thermodynamic
    properties from a cluster expansion hamiltonian. Think of this as fixed
    size supercell optimized to calculate correlation vectors and local changes
    to correlation vectors from site flips.
    """

    def __init__(self, cluster_expansion, supercell_matrix):
        pass

    def corr_from_occu(self, occu):
        pass

    def delta_corr(self, corr, flips):
        pass

    def structure_from_occu(self, occu):
        pass

    def encode_occu(self, occu):
        pass

    def decode_occu(self, encoded_occu):
        pass

    def from_dict(cls, d):
        pass

    def as_dict(self) -> dict:
        pass
