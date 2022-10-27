"""A few general utilities functions."""

__author__ = "Luis Barroso-Luque"

import inspect
import re
import warnings
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
        if inspect.isabstract(sub_class):
            sub_classes.update(get_subclasses(sub_class))
        else:
            sub_classes[sub_class.__name__] = sub_class
    return sub_classes


def progress_bar(display, total, description):
    """Get a tqdm progress bar interface.

    If the tqdm library is not installed, this will be an empty progress bar
    that does nothing.

    Args:
        display (bool):
            if true, a real progress bar will be returned.
        total (int):
            the total size of the progress bar.
        description (str):
            description to print in progress bar.
    """
    try:
        # pylint: disable=import-outside-toplevel
        import tqdm
    except ImportError:
        tqdm = None

    if display:
        if tqdm is None:
            warnings.warn(
                "tqdm library needs to be installed to show a " " progress bar."
            )
            return _EmptyBar()

        return tqdm.tqdm(total=total, desc=description)
        # if display is True:
        #   return tqdm.tqdm(total=total, desc=description)
        # else:
        #    return getattr(tqdm, "tqdm_" + display)(total=total)
    return _EmptyBar()


class _EmptyBar:
    """A dummy progress bar.

    Idea take from emce:
    https://github.com/dfm/emcee/blob/main/src/emcee/pbar.py
    """

    # pylint: disable=missing-function-docstring

    def __init__(self):
        pass

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def update(self, *args):
        pass
