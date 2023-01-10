"""Special Quasi-Random Structures (SQS)."""

__author__ = "Luis Barroso-Luque"


import warnings
from abc import ABC, abstractmethod
from collections import namedtuple

import numpy as np

from smol._utils import derived_class_factory
from smol.capp import enumerate_supercell_matrices, gen_random_ordered_occupancy
from smol.cofe import ClusterSubspace
from smol.moca import Ensemble, SampleContainer, Sampler
from smol.moca.kernel import MulticellMetropolis, mckernel_factory
from smol.moca.kernel._trace import Trace
from smol.moca.processor import FeatureDistanceProcessor

SQS = namedtuple("SQS", ["score", "features", "supercell_matrix", "structure"])


class SQSGenerator(ABC):
    """Abstract SQSGenerator class."""

    def __init__(
        self,
        cluster_subspace,
        supercell_size,
        feature_type="correlation",
        target_vector=None,
        target_weights=None,
        match_weight=1.0,
        match_tol=1e-5,
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
            target_vector (ndarray): optional
                target feature vector to use for distance calculations
            target_weights (ndarray): optional
                weights for each target feature
            match_weight (float): optional
                weight for the in the wL term above. That is how much to weight the
                largest diameter below which all features are matched exactly.
                Set to any number >0 to use this term. Default is 1.0. to ignore it
                set it to zero.
            match_tol (float): optional
                tolerance for matching features. Default is 1e-5.
            supercell_matrices (list of ndarray): optional
                list of supercell matrices to use. If None, all symmetrically distinct
                supercell matrices are generated.
        """
        self.cluster_subspace = cluster_subspace
        self.supercell_size = supercell_size

        if feature_type == "correlation":
            num_features = len(cluster_subspace) - 1  # remove constant
        else:
            raise ValueError(f"feature_type {feature_type} not supported")

        if target_weights is None:
            target_weights = np.ones(num_features)
        else:
            if len(target_weights) != num_features:
                raise ValueError(
                    "target_feature_weights must be of length {num_features}"
                )

        if target_vector is None:
            target_vector = np.zeros(num_features)
        else:
            if len(target_vector) != num_features:
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
            supercell_matrices = enumerate_supercell_matrices(
                supercell_size, cluster_subspace.symops
            )

        self._processors = [
            derived_class_factory(
                feature_type.capitalize() + "DistanceProcessor",
                FeatureDistanceProcessor,
                cluster_subspace,
                scm,
                target_vector=target_vector,
                target_weights=target_weights,
                match_weight=match_weight,
                match_tol=match_tol,
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
        match_weight=1.0,
        match_tol=1e-5,
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
            match_weight (float): optional
                weight for the in the wL term above. That is how much to weight the
                largest diameter below which all features are matched exactly.
                Set to any number >0 to use this term. Default is 1.0. to ignore it
                set it to zero.
            match_tol (float): optional
                tolerance for matching features. Default is 1e-5.
            supercell_matrices (list of ndarray): optional
                list of supercell matrices to use. If None, all symmetrically distinct
                supercell matrices are generated.
            **kwargs:
                additional keyword arguments specific to the SQSGenerator

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
            match_weight=match_weight,
            match_tol=match_tol,
            supercell_matrices=supercell_matrices,
            **kwargs,
        )

    @property
    @abstractmethod
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


class StochasticSQSGenerator(SQSGenerator):
    """StochasticSQSGenertor class.

    Generate special quasirandom structures using simulated annealing, based on the
    following work

    https://doi.org/10.1016/j.calphad.2013.06.006
    """

    def __init__(
        self,
        cluster_subspace,
        supercell_size,
        feature_type="correlation",
        target_feature_vector=None,
        target_feature_weights=None,
        match_weight=1.0,
        match_tol=1e-5,
        supercell_matrices=None,
        kernel_kwargs=None,
        **kwargs,
    ):
        """Initialize StochasticSQSGenerator.

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
            match_weight (float): optional
                weight for the in the wL term above. That is how much to weight the
                largest diameter below which all features are matched exactly.
                Set to any number >0 to use this term. Default is 1.0. to ignore it
                set it to zero.
            match_tol (float): optional
                tolerance for matching features. Default is 1e-5.
            supercell_matrices (list of ndarray): optional
                list of supercell matrices to use. If None, all symmetrically distinct
                supercell matrices are generated.
            kernel_kwargs (dict): optional
                keyword arguments for the transition kernel used in each supercell.
                For example see the documentation for :class:`Metropolis` kernel.
            **kwargs:
                keyword arguments used to initialize the :class:`MulticellMetropolis`
                kernel to be used for the simulated annealing.
        """
        super().__init__(
            cluster_subspace,
            supercell_size,
            feature_type,
            target_feature_vector,
            target_feature_weights,
            match_weight,
            match_tol,
            supercell_matrices,
        )
        step_type = kwargs.pop("step_type", "swap")
        temperature = kwargs.pop("temperature", 5.0)

        if step_type != "swap":
            warnings.warn(
                f"Step type {step_type} was provided.\n Swap steps are recommended, make sure this is what you want.",
                UserWarning,
            )

        kernel_kwargs = kernel_kwargs or {}
        kernels = [
            mckernel_factory(
                "metropolis",
                Ensemble(processor),
                step_type,
                temperature=temperature,
                **kernel_kwargs,
            )
            for processor in self._processors
        ]
        for kernel in kernels:
            kernel.kB = 1.0  # set kB to 1.0 units
        self._kernel = MulticellMetropolis(kernels, temperature=temperature, **kwargs)
        self._kernel.kB = 1.0  # set kB to 1.0 units

        # get a trial trace to initialize sample container trace
        _trace = self._kernel.compute_initial_trace(
            np.zeros(kernels[0].ensemble.num_sites, dtype=int)
        )
        sample_trace = Trace(
            **{
                name: np.empty((0, 1, *value.shape), dtype=value.dtype)
                for name, value in _trace.items()
            }
        )
        # TODO refactor sample container to accept tuple of sublattices and just take the ensemble
        container = SampleContainer(
            kernels[0].ensemble.sublattices,
            kernels[0].ensemble.natural_parameters,
            kernels[0].ensemble.num_energy_coefs,
            sample_trace,
        )
        self._sampler = Sampler([self._kernel], container)

    def num_structures(self):
        """Get number of structures generated."""
        return self._sampler.samples.num_samples

    # TODO sampler.run with initial occus leads to overflow, corrs grow like crazy
    # running from last trace leads to crash (sigkill)
    def generate(
        self, mcmc_steps, initial_occupancies=None, temperatures=None, **kwargs
    ):
        """Generate a SQS structures.

        Args:
            mcmc_steps (int):
                number of mcmc steps per temperature to perfurm simulated annealing
            initial_occupancies (ndarray): optional
                initial occupancies to use in the simulated anneal search, an occupancy
                must be given for each of the supercell shapes, and these occupancies
                must have the correct compositions.
            temperatures (list of float): optional
                temperatures to use for the mcmc steps, temperatures are unitless
                the recommended range should be only single digits.
            **kwargs:
                additional keyword arguments for the simulated annealing. See the anneal
                method in the :class:`Sampler`.
        """
        if initial_occupancies is None and self._sampler.samples.num_samples == 0:
            initial_occupancies = self._get_initial_occupancies()
        elif initial_occupancies is not None:
            shape = (len(self._processors), self._kernel.ensemble._num_sites)
            if initial_occupancies.shape != shape:
                raise ValueError(
                    f"initial occupancies must be of shape {shape}, got {initial_occupancies.shape}"
                )

        if temperatures is None:
            temperatures = np.linspace(5.0, 0.01, 20)  # TODO benchmark this

        # TODO need to have a way to set the initial occupancies for all the kernels
        #  here....
        self._sampler.anneal(
            temperatures, mcmc_steps, initial_occupancies=initial_occupancies, **kwargs
        )

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

    def _get_initial_occupancies(self):
        """Get initial occupancies for the simulated annealing.

        Gets initial occupances for each supercell shape at the composition of the
        distordered structure in the given cluster subspace.

        Returns:
            ndarray of occupancies
        """
        occupancies = np.vstack(
            [
                gen_random_ordered_occupancy(
                    proc, composition=self.cluster_subspace.structure.composition
                )
                for proc in self._processors
            ]
        )
        return occupancies
