from abc import ABCMeta, abstractmethod

from monty.json import MSONable


class BaseProcessor(MSONable, metaclass=ABCMeta):
    """Abstract base class for processors.

    A processor is used to provide a quick way to calculated energy differences
    (probability ratio's) between two adjacent configurational states.
    """

    @abstractmethod
    def compute_property_change(self, occu, flips):
        """Compute change in property from a set of flips.

        Args:
            occu (ndarray):
                encoded occupancy array
            flips (list):
                list of tuples for (index of site, specie code to set)

        Returns:
            float:  property difference between inital and final states
        """
        return

    def as_dict(self) -> dict:
        """
        Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        d = {'@module': self.__class__.__module__,
             '@class': self.__class__.__name__}
        return d

    @classmethod
    def from_dict(cls, d):
        """Create a CEProcessor from serialized MSONable dict."""
        # is this good design?
        try:
            for derived in cls.__subclasses__():
                if derived.__name__ == d['@class']:
                    return derived.from_dict(d)
        except KeyError:
            raise NameError(f"Unable to instantiate {d['@class']}.")