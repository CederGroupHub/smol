"""Special Quasi-Random Structures (SQS)."""

__author__ = "Luis Barroso-Luque"


from abc import ABC, abstractmethod
from collections import namedtuple

import numpy as np

from smol._utils import derived_class_factory
from smol.capp.tools import gen_supercell_matrices
from smol.cofe import ClusterSubspace
from smol.moca.processor import FeatureDistanceProcessor

SQS = namedtuple("SQS", ["score", "features", "supercell_matrix", "structure"])


class SQSGenerator(ABC):
    """Abstract SQSGenerator class."""

    def __init__(
        self,
        cluster_subspace,
        supercell_size,
        feature_type="correlation",
        target_feature_vector=None,
        target_feature_weights=None,
        supercell_matrices=None,
        **kwargs,
    ):
        """Initialize SQSGenerator.

        Args:
            cluster_subspace (ClusterSubspace):
                cluster subspace used to determine features
            supercell_size (int):
                size of the supercell in multiples of the primitive cell
            feature_type (str): optional
                type of features to be used to determine SQS.
                options are: "correlation"
            target_feature_vector (ndarray): optional
                target feature vector to use for distance calculations
            target_feature_weights (ndarray): optional
                weights for each target feature
            supercell_matrices (list of ndarray): optional
                list of supercell matrices to use
            **kwargs:
                additional keyword arguments to be passed to the FeatureDistanceProcessor
                initialization
        """
        self.cluster_subspace = cluster_subspace
        self.supercell_size = supercell_size

        if feature_type == "correlation":
            num_features = len(cluster_subspace)
        else:
            raise ValueError(f"feature_type {feature_type} not supported")

        if target_feature_weights is None:
            target_feature_weights = np.ones(num_features)
        else:
            if len(target_feature_weights) != num_features:
                raise ValueError(
                    "target_feature_weights must be of length {num_features}"
                )

        if target_feature_vector is None:
            target_feature_vector = np.zeros(num_features)
        else:
            if len(target_feature_vector) != num_features:
                raise ValueError(
                    "target_feature_vector must be of length {num_features}"
                )

        if supercell_matrices is not None:
            for scm in supercell_matrices:
                if scm.shape != (3, 3):
                    raise ValueError("supercell matrices must be 3x3")
                if np.linalg.det(scm) != supercell_size:
                    raise ValueError(
                        "supercell matrices must have determinant equal to supercell_size"
                    )
        else:
            supercell_matrices = gen_supercell_matrices(
                supercell_size, cluster_subspace.symmops
            )

        self._processors = [
            derived_class_factory(
                feature_type,
                FeatureDistanceProcessor,
                cluster_subspace,
                scm,
                target_feature_vector=target_feature_vector,
                target_feature_weights=target_feature_weights,
                **kwargs,
            )
            for scm in supercell_matrices
        ]

    @classmethod
    def from_structure(
        cls,
        structure,
        cutoffs,
        supercell_size,
        basis="indicator",
        orthonormal=True,
        use_concentration=True,
        feature_type="correlation",
        target_feature_vector=None,
        target_feature_weights=None,
        supercell_matrices=None,
        **kwargs,
    ):
        """Initialize an SQSGenerater from a disordered structure.

        Args:
            structure (Structure):
                a disordered pymatgen structure
            cutoffs (dict of int: float):
                cluster diameter cutoffs
            supercell_size (int):
                size of the supercell in multiples of the primitive cell
            basis (str):
                a string specifying the site basis functions
            orthonormal (bool):
                whether to enforce an orthonormal basis. From the current
                available bases only the indicator basis is not orthogonal out
                of the box
            use_concentration (bool):
                if True, the concentrations in the prim structure sites will be
                used to orthormalize site bases. This gives a cluster
                subspace centered about the prim composition.
            feature_type (str): optional
                type of features to be used to determine SQS.
                options are: "correlation"
            target_feature_vector (ndarray): optional
                target feature vector to use for distance calculations
            target_feature_weights (ndarray): optional
                weights for each target feature
            supercell_matrices (list of ndarray): optional
                list of supercell matrices to use
            **kwargs:
                additional keyword arguments to be passed to the FeatureDistanceProcessor
                initialization

        Returns:
            SQSGenerator
        """
        subspace = ClusterSubspace.from_cutoffs(
            structure,
            cutoffs,
            basis=basis,
            orthonormal=orthonormal,
            use_concentration=use_concentration,
        )
        return cls(
            subspace,
            supercell_size,
            feature_type=feature_type,
            target_feature_vector=target_feature_vector,
            target_feature_weights=target_feature_weights,
            supercell_matrices=supercell_matrices,
            **kwargs,
        )

    @abstractmethod
    @property
    def num_structures(self):
        """Get number of structures generated."""

    def calculate_score(self, structure):
        """Calculate the SQS score for a structure.

        Args:
            structure (Structure):
                a pymatgen ordered structure

        Returns:
            float: the SQS score
        """
        raise NotImplementedError("calculate_score not implemented")

    @abstractmethod
    def generate(self, *args, **kwargs):
        """Generate a SQS structures."""
        return

    @abstractmethod
    def get_best_sqs(
        self, num_structures=1, remove_duplicates=True, reduction_algorithm=None
    ):
        """Get the best SQS structures.

        generate method must be called, otherwise no SQS will be returned.

        Args:
            num_structures (int): optional
                number of structures to return
            remove_duplicates (bool): optional
                whether to remove duplicate structures
            reduction_algorithm (callable): optional
                function to reduce the number of structures returned
                should take a list of SQS structures and return a list of SQS structures

        Returns:
            list of SQS : list ranked by SQS score
        """
        if num_structures > self.num_structures:
            raise ValueError(
                f"num_structures must be less than or equal to the total number of "
                f"structures generated {self.num_structures}."
            )
        return


# class StochasticSQSGenerator(SQSGenerator):
#    pass
