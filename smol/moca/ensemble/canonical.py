
from smol.moca.ensemble.base import BaseEnsemble


class CanonicalEnsemble(BaseEnsemble):
    """
    A Canonical Ensemble class to run Monte Carlo Simulations
    """

    def __init__(self, processor, initial_occupancy=None, save_interval=None,
                 seed=None):
        """
        Args:
            processor (Processor Class):
                A processor that can compute the change in a property given
                a set of flips.
            inital_occupancy (array):
                Initial occupancy vector. If none is given then a random one
                will be used.
            save_interval (int):
                interval of steps to save the current occupancy and property
            seed (int):
                seed for random number generator
        """

        super().__init__(processor, initial_occupancy, save_interval, seed)