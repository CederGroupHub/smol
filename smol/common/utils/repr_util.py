"""Representation utilities."""

__author__ = "Luis Barroso-Luque"

from typing import Dict, Any

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
