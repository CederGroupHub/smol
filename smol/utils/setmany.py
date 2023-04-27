"""Implementations of general descriptors to manage attributes."""

from collections.abc import Sequence


class SetMany:
    """Descriptor to set an attributed value that cascades to a sequence of objects.

    This descriptor is useful when an attribute is needs to be synchronized between
    a class that holds a sequence of objects that have a corresponding attribute.
    """

    def __init__(self, attr_name, container_name, container_attr_name=None):
        """Define the name of the attribute and the container.

        Args:
            attr_name (str):
                name of the attribute to set.
            container_name (str):
                name of the container attribute holding instances with attributes to
                be set to the same value as the attribute being set.
            container_attr_name (str): optional
                name of the attribute in the container that holds the instances.
        """
        self._attr_name = attr_name
        self._container_name = container_name
        self._container_attr_name = (
            container_attr_name if container_attr_name else attr_name
        )

    def __get__(self, instance, objtype=None):
        """Get the value of the attribute."""
        # we don't need to get all values, and maybe we should check they are all the
        # same
        values = [
            getattr(obj, self._container_attr_name)
            for obj in self._container_objs(instance)
        ]
        return values[0]

    def __set__(self, instance, value):
        """Set the value of the attribute and cascade to the container."""
        for obj in self._container_objs(instance):
            setattr(obj, self._container_attr_name, value)

    def _container_objs(self, instance):
        """Yield the values in the container."""
        container = getattr(instance, self._container_name)
        if isinstance(container, Sequence):
            iterable = container
        elif hasattr(container, "values"):
            iterable = container.values()
        else:
            raise TypeError(
                "container must be an iterable or a dict-like object, not {}".format(
                    type(container)
                )
            )

        yield from iterable
