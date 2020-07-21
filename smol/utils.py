"""A few random utilities that have no place to go."""

__author__ = "Luis Barroso-Luque"

import inspect
from typing import Dict, Any, List


def _repr(instance: object, **fields: Dict[str, Any]) -> str:
    """Create object representation.

    A helper function for repr overloading in classes.
    """
    attrs = []

    for key, field in fields.items():
        attrs.append(f'{key}={field!r}')

    if len(attrs) == 0:
        return f"<{instance.__class__.__name__}" \
               f"{hex(id(instance))}({','.join(attrs)})>"
    else:
        return f"<{instance.__class__.__name__} {hex(id(instance))}>"


def _get_subclasses(base_class: object) -> List[object]:
    """Get all non-abstract subclasses of a class.

    Gets all non-abstract classes that inherit from the given base class in
    a module. This is used to obtain all the available basis functions.
    """
    sub_classes = []
    for c in base_class.__subclasses__():
        if inspect.isabstract(c):
            sub_classes += _get_subclasses(c)
        else:
            sub_classes.append(c.__name__[:-5].lower())
    return sub_classes
