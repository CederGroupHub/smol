"""Some SimpleNamespace subclasses for use in the MCKernels, MCUshers and MCBias classes."""

__author__ = "Luis Barroso-Luque"

from types import SimpleNamespace

import numpy as np
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


class Trace(SimpleNamespace):
    """Simple Trace class.

    A Trace is a simple namespace to hold states and values to be recorded
    during MC sampling.
    """

    def __init__(self, /, **kwargs):  # noqa
        if not all(isinstance(val, np.ndarray) for val in kwargs.values()):
            raise TypeError("Trace only supports attributes of type ndarray.")
        super().__init__(**kwargs)

    @property
    def names(self):
        """Get all attribute names."""
        return tuple(self.__dict__.keys())

    def items(self):
        """Return generator for (name, attribute)."""
        yield from self.__dict__.items()

    def __setattr__(self, name, value):
        """Set only ndarrays as attributes."""
        if isinstance(value, (float, int)):
            value = np.array([value])

        if not isinstance(value, np.ndarray):
            raise TypeError("Trace only supports attributes of type ndarray.")
        self.__dict__[name] = value

    def as_dict(self):
        """Return copy of underlying dictionary."""
        return self.__dict__.copy()


class StepTrace(Trace):
    """StepTrace class.

    Same as Trace above but holds a default "delta_trace" inner trace to hold
    trace values that represent changes from previous values, to be handled
    similarly to delta_features and delta_energy.

    A StepTrace object is set as an MCKernel's attribute to record
    kernel specific values during sampling.
    """

    def __init__(self, /, **kwargs):  # noqa
        super().__init__(**kwargs)
        super(Trace, self).__setattr__("delta_trace", Trace())

    @property
    def names(self):
        """Get all field names. Removes delta_trace from field names."""
        return tuple(name for name in super().names if name != "delta_trace")

    def items(self):
        """Return generator for (name, attribute). Skips delta_trace."""
        for name, value in self.__dict__.items():
            if name == "delta_trace":
                continue
            yield name, value

    def __setattr__(self, name, value):
        """Set only ndarrays as attributes."""
        if name == "delta_trace":
            raise ValueError("Attribute name 'delta_trace' is reserved.")
        if not isinstance(value, np.ndarray):
            raise TypeError("Trace only supports attributes of type ndarray.")
        self.__dict__[name] = value

    def as_dict(self):
        """Return copy of serializable dictionary."""
        step_trace_d = self.__dict__.copy()
        step_trace_d["delta_trace"] = step_trace_d["delta_trace"].as_dict()
        return step_trace_d
