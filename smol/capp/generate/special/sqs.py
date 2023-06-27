"""Special Quasi-Random Structures (SQS)."""

__author__ = "Luis Barroso-Luque"


import warnings
from abc import ABC, abstractmethod
from collections import deque, namedtuple
from copy import deepcopy

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher

from smol.capp.generate.enumerate import enumerate_supercell_matrices
from smol.capp.generate.random import generate_random_ordered_occupancy
from smol.cofe import ClusterSubspace
from smol.moca import Ensemble, SampleContainer, Sampler
from smol.moca.kernel import MulticellMetropolis, mckernel_factory
from smol.moca.processor.distance import DistanceProcessor
from smol.moca.trace import Trace
from smol.utils.class_utils import class_name_from_str, derived_class_factory
from smol.utils.progressbar import progress_bar

SQS = namedtuple("SQS", ["structure", "score", "feature_distance", "supercell_matrix"])


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
                cluster subspace used to determine feature distance
            supercell_size (int):
                size of the supercell in multiples of the primitive cell
            feature_type (str): optional
                type of features to be used to determine SQS.
                options are "correlation" and "cluster-interaction".
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
        self._sqs_deque = None

        if feature_type == "correlation":
            num_features = len(cluster_subspace)
        elif feature_type == "cluster-interaction":
            num_features = cluster_subspace.num_orbits
        else:
            raise ValueError(
                f"feature_type {feature_type} not supported.\n "
                f"Valid options are: 'correlation', 'cluster-interaction'"
            )

        if target_weights is None:
            target_weights = np.ones(num_features - 1)  # remove constant
        else:
            if len(target_weights) != num_features - 1:
                raise ValueError(f"target_weights must be of length {num_features - 1}")

        if target_vector is None:
            target_vector = np.zeros(num_features)
        else:
            if len(target_vector) != num_features:
                raise ValueError(
                    f"target feature vector must be of length {num_features}"
                )

        if supercell_matrices is not None:
            for scm in supercell_matrices:
                if scm.shape != (3, 3):
                    raise ValueError("supercell matrices must be 3x3")
                if not np.isclose(np.linalg.det(scm), supercell_size):
                    raise ValueError(
                        "supercell matrices must have determinant equal to "
                        "supercell_size"
                    )
        else:
            supercell_matrices = enumerate_supercell_matrices(
                supercell_size, cluster_subspace.symops
            )
            # reverse since most skewed cells are enumerated first
            supercell_matrices.reverse()

        self._processors_by_scm = {
            tuple(sorted(tuple(s.tolist()) for s in scm)): derived_class_factory(
                class_name_from_str(feature_type + "DistanceProcessor"),
                DistanceProcessor,
                cluster_subspace,
                scm,
                target_vector=target_vector,
                target_weights=target_weights,
                match_weight=match_weight,
                match_tol=match_tol,
            )
            for scm in supercell_matrices
        }
        self._processors = list(self._processors_by_scm.values())

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
        target_vector=None,
        target_weights=None,
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
                available bases only the indicator basis is not orthogonal
                out-of-the-box
            use_concentration (bool):
                if True, the concentrations in the prim structure sites will be
                used to orthormalize site bases. This gives a cluster
                subspace centered about the prim composition.
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
                tolerance for exact matching features. Default is 1e-5.
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
            target_vector=target_vector,
            target_weights=target_weights,
            match_weight=match_weight,
            match_tol=match_tol,
            supercell_matrices=supercell_matrices,
            **kwargs,
        )

    @property
    def num_structures(self):
        """Get number of structures generated."""
        return 0 if self._sqs_deque is None else len(self._sqs_deque)

    @abstractmethod
    def generate(self, *args, **kwargs):
        """Generate a SQS structures."""
        return

    def compute_score(self, structure, supercell_matrix=None):
        """Compute the SQS score for a structure.

        Args:
            structure (Structure):
                ordered structure to compute SQS score for
            supercell_matrix (ndarray): optional
                supercell matrix relating given ordered structure to the disordered
                structure.

        Returns:
            float : SQS score
        """
        processor = self._get_structure_processor(structure, supercell_matrix)
        occu = processor.occupancy_from_structure(structure)
        return processor.compute_property(occu)

    def compute_feature_distance(self, structure, supercell_matrix=None):
        """Compute feature distance to target vector for a given structure.

        Args:
            structure (Structure):
                ordered structure to compute SQS score for
            supercell_matrix (ndarray): optional
                supercell matrix relating given ordered structure to the disordered
                structure.

        Returns:
            float : feature distance
        """
        processor = self._get_structure_processor(structure, supercell_matrix)
        occu = processor.occupancy_from_structure(structure)
        return processor.compute_feature_vector(occu)

    def _get_structure_processor(self, structure, supercell_matrix):
        """Check if an ordered structure and supercell matrix are valid."""
        if supercell_matrix is None:
            supercell_matrix = self.cluster_subspace.scmatrix_from_structure(structure)

        if not np.isclose(np.linalg.det(supercell_matrix), self.supercell_size):
            raise ValueError(
                "Invalid supercell matrix, must have determinant equal to supercell_size"
            )
        scm = tuple(sorted(tuple(s.tolist()) for s in supercell_matrix))
        return self._processors_by_scm[scm]

    def get_best_sqs(
        self, num_structures=1, remove_duplicates=True, reduction_algorithm="niggli"
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
                options are "niggli" and "LLL".
                See documentation for Structure.get_reduced_structure in pymatgen
                https://pymatgen.org/pymatgen.core.structure.html

        Returns:
            list of SQS : list ranked by SQS score
        """
        if num_structures > self.num_structures:
            warnings.warn(
                f"num_structures is greater than the total number of "
                f"structures generated {self.num_structures}. \n Will return at most "
                f"{self.num_structures} structures."
            )

        best_traces = sorted(self._sqs_deque, key=lambda x: x.enthalpy)

        # package traces into SQS objects
        best_sqs = []
        for trace in best_traces:
            i = np.argmin(trace.enthalpy)  # in case anyone runs multiple walkers
            processor = self._processors[trace.kernel_index[i]]
            structure = processor.structure_from_occupancy(trace.occupancy[i])
            sqs = SQS(
                structure=structure,
                score=trace.enthalpy[i][0],
                feature_distance=trace.features[i],
                supercell_matrix=processor.supercell_matrix,
            )
            best_sqs.append(sqs)

            if num_structures == 1:  # just do one
                break

        if remove_duplicates:
            matcher = StructureMatcher()
            grouped_structures = matcher.group_structures(
                [sqs.structure for sqs in best_sqs]
            )
            unique_structures = [group[0] for group in grouped_structures]
            best_sqs = list(
                filter(lambda x: x.structure in unique_structures, best_sqs)
            )

        if len(best_sqs) < num_structures:
            warnings.warn(
                f"Fewer SQS structures than requested were obtained after removing "
                f"duplicates. Only {len(best_sqs)} distinct SQS will be returned."
            )
        else:
            best_sqs = best_sqs[:num_structures]

        if reduction_algorithm is not None:
            for i, sqs in enumerate(best_sqs):
                structure = sqs.structure.get_reduced_structure(
                    reduction_algo=reduction_algorithm
                )
                reduced_sqs = SQS(
                    structure=structure,
                    score=sqs.score,
                    feature_distance=sqs.feature_distance,
                    supercell_matrix=sqs.supercell_matrix,
                )
                best_sqs[i] = reduced_sqs

        return best_sqs


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
        target_vector=None,
        target_weights=None,
        match_weight=1.0,
        match_tol=1e-5,
        supercell_matrices=None,
        kernel_kwargs=None,
        **kwargs,
    ):
        """Initialize StochasticSQSGenerator.

        Args:
            cluster_subspace (ClusterSubspace):
                cluster subspace used to determine feature_distance
            supercell_size (int):
                size of the supercell in multiples of the primitive cell
            feature_type (str): optional
                type of feature_distance to be used to determine SQS.
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
            target_vector,
            target_weights,
            match_weight,
            match_tol,
            supercell_matrices,
        )
        step_type = kwargs.pop("step_type", "swap")
        temperature = kwargs.pop("temperature", 5.0)

        if step_type != "swap":
            warnings.warn(
                f"Step type {step_type} was provided.\n Swap steps are recommended, "
                f"make sure this is what you want.",
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

        container = SampleContainer(kernels[0].ensemble, sample_trace)
        container.metadata.type = "SQS-SampleContainer"
        self._sampler = Sampler([self._kernel], container)

    @property
    def sampler(self):
        """Sampler: sampler used to generate SQS."""
        return self._sampler

    def generate(
        self,
        mcmc_steps,
        temperatures=None,
        initial_occupancies=None,
        clear_previous=True,
        max_save_num=None,
        progress=False,
    ):
        """Generate a SQS structures.

        Args:
            mcmc_steps (int):
                number of mcmc steps per temperature to perform simulated annealing
            temperatures (list of float): optional
                temperatures to use for the mcmc steps, temperatures are unit-less
                the recommended range should be only single digits.
            initial_occupancies (ndarray): optional
                initial occupancies to use in the simulated anneal search, an occupancy
                must be given for each of the supercell shapes, and these occupancies
                must have the correct compositions.
            clear_previous (bool): optional
                whether to clear previous samples.
            max_save_num (int): optional
                max number of best structures to save, if None a maximum 1% of
                structures are saved. A structure is only saved if the SQS score is
                better than the previous best.
            progress (bool):
                if true will show a progress bar.
        """
        if initial_occupancies is None:
            if self._sampler.samples.num_samples == 0:
                initial_occupancies = self._get_initial_occupancies()
            else:
                initial_occupancies = self._sampler.samples.get_occupancies(flat=False)[
                    -1
                ]
        elif initial_occupancies is not None:
            shape = (len(self._processors), self._kernel.ensemble.num_sites)
            if initial_occupancies.shape != shape:
                raise ValueError(
                    f"initial occupancies must be of shape {shape}, "
                    f"got {initial_occupancies.shape}"
                )
            initial_occupancies = initial_occupancies.copy()

        max_save_num = max_save_num or max(int(0.01 * mcmc_steps), 1)
        if clear_previous:
            self._sampler.clear_samples()
            self._sqs_deque = deque(maxlen=max_save_num)
        else:
            self._sqs_deque = deque(
                self._sqs_deque, maxlen=len(self._sqs_deque) + max_save_num
            )

        if temperatures is None:
            temperatures = np.linspace(5.0, 0.01, 20)

        self._kernel.temperature = temperatures[0]
        for kernel in self._kernel.mckernels:
            kernel.temperature = temperatures[0]

        best_score = np.inf
        for trace in self._sample_sqs(
            mcmc_steps, initial_occupancies, progress=progress
        ):
            if np.any(trace.enthalpy < best_score):
                best_score = trace.enthalpy.min()
                self._sqs_deque.append(deepcopy(trace))

        for temperature in temperatures[1:]:
            self._kernel.temperature = temperature
            for kernel in self._kernel.mckernels:
                kernel.temperature = temperature
            initial_occupancies = trace.occupancy
            for trace in self._sample_sqs(
                mcmc_steps, initial_occupancies, progress=progress
            ):
                if np.any(trace.enthalpy < best_score):
                    best_score = trace.enthalpy.min()
                    self._sqs_deque.append(deepcopy(trace))

        self._sampler.samples.allocate(max_save_num)
        for trace in self._sqs_deque:
            self._sampler.samples.save_sampled_trace(trace, 1)
        self._sampler.samples.vacuum()  # vacuum since not all space may have been used

    def _sample_sqs(self, nsteps, initial_occupancies, progress=False):
        """Sample SQS structures.

        Very similar to sampler.sample but without thinning and only yielding better
        SQS structures.

        Args:
            nsteps (int):
                number of steps to sample
            progress (bool):
                whether to show a progress bar

        Yields:
            tuple: accepted, occupancies, features change, enthalpies change
        """
        occupancies, trace = self._sampler.setup_sample(initial_occupancies)
        chains, nsites = self._sampler.samples.shape
        desc = f"Generating SQS using {chains} chain(s) from cells with {nsites} sites"
        with progress_bar(progress, total=nsteps, description=desc) as p_bar:
            for _ in range(nsteps):
                # the occupancies are modified in place at each step
                for i, strace in enumerate(
                    map(
                        lambda kernel, occu: kernel.single_step(occu),
                        self._sampler.mckernels,
                        occupancies,
                    )
                ):
                    for name, value in strace.items():
                        val = getattr(trace, name)
                        val[i] = value
                        # this will mess up recording values for > 1 walkers
                        # setattr(trace, name, value)
                    if strace.accepted:
                        for name, delta_val in strace.delta_trace.items():
                            val = getattr(trace, name)
                            val[i] += delta_val
                p_bar.update()
                yield trace

    def _get_initial_occupancies(self):
        """Get initial occupancies for the simulated annealing.

        Gets initial occupancies for each supercell shape at the composition of the
        disordered structure in the given cluster subspace.

        Returns:
            ndarray of occupancies
        """
        # all processors should have the same compositions so just use one processor
        compositions = [sl.composition for sl in self._processors[0].get_sublattices()]
        occupancies = np.vstack(
            [
                generate_random_ordered_occupancy(proc, composition=compositions)
                for proc in self._processors
            ]
        )
        return occupancies
