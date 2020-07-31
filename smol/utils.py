"""A few random utilities that have no place to go."""

__author__ = "Luis Barroso-Luque"

import inspect
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


def derived_class_factory(class_name: str, base_class: object,
                          *args: Any, **kwargs: Any) -> object:
    """Return an instance of derived class from a given basis class.

    Args:
        class_name (str):
            name of class
        base_class (obect):
            base class of derived class sought
        *args:
            positional arguments for class constructor
        **kwargs:
            keyword arguments for class constructor

    Returns:
        object: instance of class with corresponding constructor args, kwargs
    """
    try:
        derived_class = get_subclasses(base_class)[class_name]
        instance = derived_class(*args, **kwargs)
    except KeyError:
        raise NotImplementedError(f'{class_name} is not implemented.')
    return instance


def get_subclasses(base_class: object) -> Dict[str, object]:
    """Get all non-abstract subclasses of a class.

    Gets all non-abstract classes that inherit from the given base class in
    a module. This is used to obtain all the available basis functions.
    """
    sub_classes = {}
    for sub_class in base_class.__subclasses__():
        if inspect.isabstract(sub_class):
            sub_classes.update(get_subclasses(sub_class))
        else:
            sub_classes[sub_class.__name__] = sub_class
    return sub_classes
