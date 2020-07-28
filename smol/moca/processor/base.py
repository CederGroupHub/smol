"""Implementation of base processor classes for a fixed size super cell.

Processor classes are used to represent a configuration domain for a fixed
sized supercell and should implement a "fast" way to compute the property
they represent or changes in said property from site flips. Things necessary
to run Monte Carlo sampling.

Processor classes should inherit from the BaseProcessor class. Processors
can be combined into composite processor for mixed models.
"""

__author__ = "Luis Barroso-Luque"

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import numpy as np

from pymatgen import Structure, PeriodicSite
from monty.json import MSONable
from smol.cofe.configspace.basis import get_site_spaces, get_allowed_species


class Processor(MSONable, metaclass=ABCMeta):
    """Abstract base class for processor.

    A processor is used to provide a quick way to calculated energy differences
    (probability ratio's) between two adjacent configurational states for a
    fixed system size/supercell.

    Attributes:
        unique_site_spaces (tuple):
            Tuple of all the distinct site spaces.
        allowed_species (list):
            A list of tuples of the allowed species at each site.
        size (int):
            Number of prims in the supercell structure.
    """
    def __init__(self, cluster_subspace, supercell_matrix, coefficients):
        """Initialize a BaseProcessor.

        Args:
            cluster_subspace (ClusterSubspace):
                A cluster subspace
            supercell_matrix (ndarray):
                An array representing the supercell matrix with respect to the
                Cluster Expansion prim structure.
            coefficients:
                single or array of fit coefficients.
        """
        self._subspace = cluster_subspace
        self._structure = self._subspace.structure.copy()
        self._structure.make_supercell(supercell_matrix)
        self._scmatrix = supercell_matrix

        self.coefs = coefficients
        # this can be used (maybe should) to check if a flip is valid
        site_spaces = get_site_spaces(self._subspace.expansion_structure)
        self.unique_site_spaces = tuple(OrderedDict(space) for space in
                                        set(tuple(spaces.items())
                                            for spaces in site_spaces))

        self.allowed_species = get_allowed_species(self.structure)
        self.size = self._subspace.num_prims_from_matrix(supercell_matrix)

    @property
    def cluster_subspace(self):
        """Get the underlying cluster subspace."""
        return self._subspace

    @property
    def structure(self):
        """Get the underlying supercell disordered structure."""
        return self._structure

    @property
    def num_sites(self):
        """Get total number of sites in supercell."""
        return len(self.cluster_subspace.structure) * self.size

    @property
    def supercell_matrix(self):
        """Get the give supercell matrix."""
        return self._scmatrix

    @abstractmethod
    def compute_feature_vector(self, occupancy):
        """Compute the feature vector for a given occupancy array.

        Each entry in the correlation vector corresponds to a particular
        symmetrically distinct bit ordering.

        Args:
            occupancy (ndarray):
                encoded occupation array

        Returns:
            array: correlation vector
        """
        return

    @abstractmethod
    def compute_feature_vector_change(self, occupancy, flips):
        """
        Compute the change in the feature vector from a list of flips.

        Args:
            occupancy (ndarray):
                encoded occupancy array
            flips (list of tuple):
                list of tuples with two elements. Each tuple represents a
                single flip where the first element is the index of the site
                in the occupancy array and the second element is the index
                for the new species to place at that site.

        Returns:
            array: change in correlation vector
        """
        return

    def compute_property(self, occupancy):
        """Compute the value of the property for the given occupancy array.

        Args:
            occupancy (ndarray):
                encoded occupancy array
        Returns:
            float: predicted property
        """
        return np.dot(self.coefs, self.compute_feature_vector(occupancy))

    def compute_property_change(self, occupancy, flips):
        """Compute change in property from a set of flips.

        Args:
            occupancy (ndarray):
                encoded occupancy array
            flips (list):
                list of tuples for (index of site, specie code to set)

        Returns:
            float:  property difference between inital and final states
        """
        return np.dot(self.coefs,
                      self.compute_feature_vector_change(occupancy, flips))

    def occupancy_from_structure(self, structure):
        """Get the occupancy array for a given structure.

        The structure must strictly be a supercell of the prim according to the
        processor's supercell matrix.

        Args:
            structure (Structure):
                A pymatgen structure (related to the cluster-expansion prim
                by the supercell matrix passed to the processor)
        Returns: encoded occupancy array
            list
        """
        occu = self._subspace.occupancy_from_structure(structure,
                                                       scmatrix=self.supercell_matrix)  # noqa
        return self.encode_occupancy(occu)

    def structure_from_occupancy(self, occupancy):
        """Get pymatgen Structure from an occupancy array.

        Args:
            occupancy (ndarray):
                encoded occupancy array

        Returns:
            Structure
        """
        occupancy = self.decode_occupancy(occupancy)
        sites = []
        for sp, s in zip(occupancy, self.structure):
            if sp != 'Vacancy':
                site = PeriodicSite(sp, s.frac_coords, self.structure.lattice)
                sites.append(site)
        return Structure.from_sites(sites)

    def encode_occupancy(self, occupancy):
        """Encode occupancy array of species str to ints."""
        # TODO check if setting to np.intc improves speed
        return np.array([species.index(sp) for species, sp
                         in zip(self.allowed_species, occupancy)],
                        dtype=int)

    def decode_occupancy(self, encoded_occupancy):
        """Decode an encoded occupancy array of int to species str."""
        return [species[i] for i, species in
                zip(encoded_occupancy, self.allowed_species)]

    def get_average_drift(self, iterations=1000):
        """Get the average forward and reverse drift for the given property.

        This is a sanity check function. The drift value should be very, very,
        very small, the smaller the better...think machine precision values.

        The average drift is the difference between the quick routine for used
        for MC to get a property difference from a single flip and the
        change in that property from explicitly calculating it fully for the
        initial state and the flipped state.

        Args:
            iterations (int): optional
                number of iterations/flips to compute.

        Returns:
            tuple: (float, float) forward and reverse average property drift
        """
        forward_drift, reverse_drift = 0.0, 0.0
        trajectory = []
        occu = [np.random.choice(species) for species in self.allowed_species]
        occu = self.encode_occupancy(occu)
        for _ in range(iterations):
            site = np.random.randint(self.size)
            species = set(range(len(self.allowed_species[site])))-{occu[site]}
            species = np.random.choice(list(species))
            delta_prop = self.compute_property_change(occu, [(site, species)])
            new_occu = occu.copy()
            new_occu[site] = species
            prop = self.compute_property(occu)
            new_prop = self.compute_property(new_occu)
            forward_drift += (new_prop - prop) - delta_prop
            reverse_flips = [(site, occu[site])]
            trajectory.append((prop - new_prop, new_occu, reverse_flips))
            occu = new_occu

        forward_drift /= iterations
        reverse_drift = sum(dp - self.compute_property_change(o, f)
                            for dp, o, f in trajectory) / iterations
        return forward_drift, reverse_drift

    def as_dict(self) -> dict:
        """
        Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        d = {'@module': self.__class__.__module__,
             '@class': self.__class__.__name__,
             'cluster_subspace': self.cluster_subspace.as_dict(),
             'supercell_matrix': self.supercell_matrix.tolist(),
             'coefficients': np.array(self.coefs).tolist()}
        return d

    # TODO get rid of this and inspect in polymorphic constructors?
    @classmethod
    def from_dict(cls, d):
        """Create a processor from serialized MSONable dict."""
        # is this good design?
        try:
            for derived in cls.__subclasses__():
                if derived.__name__ == d['@class']:
                    return derived.from_dict(d)
        except KeyError:
            raise NameError(f"Unable to instantiate {d['@class']}.")
