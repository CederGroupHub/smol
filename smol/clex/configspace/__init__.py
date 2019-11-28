from .cluster import Cluster
from .orbit import Orbit
from .supercell import ClusterSupercell
from . import functionspace


def basis_factory(basis_name, *args, **kwargs):
    """Tries to return an instance of a Basis class"""
    try:
        class_name = basis_name.capitalize() + 'Basis'
        basis_class = getattr(functionspace, class_name)
        instance = basis_class(*args, **kwargs)
    except AttributeError:
        raise functionspace.BasisNotImplemented(f'{basis_name} is not implemented. '
              f'Choose one of {[c.__name__[:-5].lower() for c in functionspace.SiteBasis.__subclasses__()]}')
    return instance
