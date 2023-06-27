"""A metadata class to recorded sampling metadata in MCKernels, MCUshers, MCBias classes."""

__author__ = "Luis Barroso-Luque"

from types import SimpleNamespace

from monty.json import MontyDecoder, MSONable, jsanitize


class Metadata(SimpleNamespace, MSONable):
    """A simple namespace to hold the metadata specifications of a class.

    It is an MSONable class well to allow for easy serialization.

    This class should be used to record the specifications used to generate data, for example
    in the sampling metadata dictionary of the SampleContainer.

    Monte Carlo helper classes include:
        * MCKernels
        * MCUshers
        * MCBias
    """

    def __init__(self, cls_name=None, **kwargs):
        """Initialize the namespace.

        # TODO without default cls_name=None deepcopy fails.

        Args:
            cls_name (str):
                The name of the class for which specifications are being
                recorded.
            **kwargs:
                keyword arguments specifications.
        """
        kwargs["cls_name"] = cls_name
        super().__init__(**kwargs)

    def as_dict(self):
        """Return copy of underlying dictionary."""
        d = self.__dict__.copy()
        for k, v in d.items():
            if isinstance(v, MSONable):
                d[k] = v.as_dict()
            else:
                d[k] = jsanitize(v)
        d.update(
            {"@class": self.__class__.__name__, "@module": self.__class__.__module__}
        )
        return d

    @classmethod
    def from_dict(cls, d):
        """Initialize from dictionary."""
        # try to recreate any MSONables in the dictionary
        decoded = {
            k: MontyDecoder().process_decoded(v)
            for k, v in d.items()
            if not k.startswith("@")
        }
        return cls(**decoded)
