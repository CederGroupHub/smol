"""A few general utilities functions."""

__author__ = "Luis Barroso-Luque"

import inspect
import re
from typing import Any, Dict


def class_name_from_str(class_str):
    """Return a class name based on given string.

    Assumes all class names are properly named with Camel Caps.

    Used to generalize how classes can be identified by also allowing
    capitalized words in class name to be separated by a hyphen and also
    allowing capitalization to be ignored.

    i.e. all of the following are valid strings for class TheBestClass:
    'TheBestClass', 'The-Best-Class', 'the-best-class', 'The-best-Class', etc

    Args:
        class_str (str):
            string identifying a class name.
    Returns: str
        class name in camel caps
    """
    return "".join(
        [
            s.capitalize()
            for sub in class_str.split("-")
            for s in re.findall("[a-zA-Z][^A-Z]*", sub)
        ]
    )


def derived_class_factory(
    class_name: str, base_class: object, *args: Any, **kwargs: Any
) -> object:
    """Return an instance of derived class from a given basis class.

    Args:
        class_name (str):
            name of class
        base_class (object):
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
    except KeyError as key_error:
        raise NotImplementedError(f"{class_name} is not implemented.") from key_error
    return instance


def get_subclasses(base_class: object) -> Dict[str, object]:
    """Get all non-abstract subclasses of a class.

    Gets all non-abstract classes that inherit from the given base class in
    a module.
    """
    sub_classes = {}
    for sub_class in base_class.__subclasses__():
        sub_classes.update(get_subclasses(sub_class))

        if not inspect.isabstract(sub_class):
            sub_classes[sub_class.__name__] = sub_class

    return sub_classes
