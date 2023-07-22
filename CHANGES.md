# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# [Unreleased]
Use this section to keep track of changes in the works.
### Added
### Changed
### Fixed
### Removed
### Deprecated

# [v0.4.1](https://github.com/CederGroupHub/smol/releases/tag/v0.4.0) (2023-07-21)
### Fixed
* Fix #386 by making cython extension classes pickleable. #387 @lbluque
* Fix #385 data type issues on Windows. #392 @qchempku2017

# [v0.4.0](https://github.com/CederGroupHub/smol/releases/tag/v0.4.0) (2023-06-09)
### Added
* openmp parallelization when computing correlation and cluster interaction vectors. #338 @lbluque
* stochastic SQS generation functionality. #344 @lbluque
* enumeration of symmetrically distinct supercell matrices. #344 @lbluque
### Fixed
* Fix #334 getting subclasses of non-abstract classes. #335 @lbluque
* Fix #353 appending in `Sublattice.restricted_sites` property. #355 @qchempku2017
* Use jsanitize to serialize dicts/lists of msonables. #354 @lbluque
* Use site_mappings in `StructureWrangler.process_entry`. #363 @lbluque

# [v0.3.1](https://github.com/CederGroupHub/smol/releases/tag/v0.3.1) (2023-02-07)
### Changed
* Save ensemble as attributed in `SampleContainer` add `get_sampled_structures`. #326 @lbluque
### Fixed
* `ClusterExpansion.cluster_interaction_tensors` as a cached property and reset when pruning. #330 @lbluque
* Fix flakey unit tests. #328 @lbluque & @qchempku2017

# [v0.3.0](https://github.com/CederGroupHub/smol/releases/tag/v0.3.0) (2023-01-13)
### Added
* Cluster decomposition analysis and sampling functionality. #306 @lbluque
### Changed
* Keep ensemble as attribute in `MCKernels`. #304 @lbluque
* Change default processor into ClusterDecompositionProcessor when initializing
  Ensemble. #309 @qchempku2017
* Use -mcpu=native compile option to build successfully on newer macs. #310 @lbluque
### Fixed
* Add `polytope` and `cvxpy` to test requirements. #304 @lbluque

# [v0.2.0](https://github.com/CederGroupHub/smol/releases/tag/v0.2.0) (2022-12-11)
### Added
* `WangLandau` kernel class for density of states sampling. #294 @qchempku2017 & @lbluque
* `Metadata` class to record specifications of MC simulations. #297 @lbluque
### Fixed
*  Remove size key when all orbits of a certain size have been removed. #292 @lbluque

# [v0.1.0](https://github.com/CederGroupHub/smol/releases/tag/v0.1.0) (2022-10-20)
### Added
* Charge neutral semi grand canonical sampling. #271 @qchempku2017
* `MultiStep` and `Composite` mcushers for more flexible sampling. #279 @lbluque

# [v0.0.7](https://github.com/CederGroupHub/smol/releases/tag/v0.0.7) (2022-09-26)
### Added
* Additional tests to ensure correlation vectors are consistent across equivalent
  supercell matrices.  #262 @qchempku2017
* Unit-test updates. #269 @lbluque
### Changed
* Improved orbit alias detection. #262 @qchempku2017

# [v0.0.6](https://github.com/CederGroupHub/smol/releases/tag/v0.0.6) (2022-09-02)
:warning: This version introduced updates that change the order in which orbits are
sorted in a ClusterSubspace. This means that the order correlation functions appear
a correlation vector will be different when generating ClusterSubspaces compared to
previous versions. However, loading a ClusterSubspace from a json file that was
created with a previous version will still have its original order.
### Changed
* Include number of corr functions when sorting orbits. #256 @lbluque
* Use max distance of centroid to sites in unit cell in cluster search. #256 @lbluque
### Fixed
* Fixed search of clusters by correctly using centroid of unit cell. #255 @kamronald
### Removed
* Removed CanonicalEnsemble and SemigrandEnsemble. #257 @lbluque

# [v0.0.5](https://github.com/CederGroupHub/smol/releases/tag/v0.0.5) (2022-08-10)
### Added
* Data centering example notebook. #238 @kamronald
### Changed
* Single sampler multiple kernels. #245 @qchempku2017
### Fixed
* Fix returning all sub_orbit_mappings. #249 @lbluque

# [v0.0.4](https://github.com/CederGroupHub/smol/releases/tag/v0.0.4) (2022-06-23)
### Added
* Allow streaming to h5 in simulated annealing. #216 @lbluque
### Fixed
* Fix recording sampled traces for nwalkers > 1. #219 @lbluque
* Fix minor error in ClusterSubspace.__str__ #226 @lbluque

# [v0.0.3](https://github.com/CederGroupHub/smol/releases/tag/v0.0.3) (2022-06-03)
### Added
* Developing section of docs. #215 @lbluque
* Single `Ensemble` class for canonical and semi-grand canonical sampling. #210 @lbluque
### Changed
* Package name properly `smol` instead of `statmech-on-lattices`.
### Fixed
* Fixed #213 metadata serialization for saving of `SampleContainers`. #214 @lbluque
### Deprecated
* `SemiGrandEnsemble` and `CanonicalEnsemble`. Use `Ensemble` with or without setting
   chemical potentials instead. #210 @lbluque

# [v0.0.2](https://github.com/CederGroupHub/smol/releases/tag/v0.0.2) (2022-05-22)
### Fixed
* version dunder with pypi project rename.
* use of np.random.default_rng for reproducibility. #206 (lbluque)
* Fix passing seed explicitly in Sampler.from_ensemble

# [v0.0.1](https://github.com/CederGroupHub/smol/releases/tag/v0.0.1) (2022-04-26)
### Added
* Method to detect and identify orbit degeneracies based on supercell shape. #184 (kamronald)
* Automatic github release.
* PyPi install as `statmech-on-lattices` (arghhh)
### Changed
* Moved cython code for computing correlations to smol/correlations.pyx and imports as smol.correlations #190 (lbluque)
### Fixed
* Fix importing numpy in setup.py

# [v0.0.0](https://github.com/CederGroupHub/smol/releases/tag/v0.0.0) (2022-04-13)
#### [Full Changelog](https://github.com/CederGroupHub/smol/compare/alpha1.0.1...v0.0.0)
### Added
* `Cluster` as `pymatgen.SiteCollection`, str and repr methods for Cluster, Orbit,
  ClusterSubspace and ClusterExpansion akin to pymatgen, and functionality to render
  Clusters with crystal-toolkit. [#181](https://github.com/CederGroupHub/smol/pull/181)
  ([lbluque](https://github.com/lbluque))
*  Sublattice splitting. [#179](https://github.com/CederGroupHub/smol/pull/179)
  ([qchempku2017](https://github.com/qchempku2017))
* `StructureWrangler.get_similarity_matrix` to get similarity fractions
  between correlation vectors of training set.
  [\#153](https://github.com/CederGroupHub/smol/pull/153)
  ([kamronald](https://github.com/kamronald))
* `ClusterSubspace` with no point terms using `{1: None}`.
  [\#158](https://github.com/CederGroupHub/smol/pull/158)
  ([lbluque](https://github.com/lbluque))
* `MCBias` implementation for biased sampling, `Trace` objects for general
  state saving during sampling.
  [\#154](https://github.com/CederGroupHub/smol/pull/154)
  ([lbluque](https://github.com/lbluque))
* Active and inactive sublattices for MC sampling.
  [\#152](https://github.com/CederGroupHub/smol/pull/152)
  ([lbluque](https://github.com/lbluque))
* `SamplerContainer.to_hdf5` to save MC sample containers
[\#151](https://github.com/CederGroupHub/smol/pull/151)
  ([lbluque](https://github.com/lbluque))
* `PottsSubspace` class to generate redundant frame expansions.
[\#146](https://github.com/CederGroupHub/smol/pull/146)
  ([lbluque](https://github.com/lbluque))
* Methods `is_suborbit` and `sub_orbit_mappings` in `Orbit` and related
`function_hierarchy` and `orbit_hierarchy` in `ClusterSubspace`.
  [\#141](https://github.com/CederGroupHub/smol/pull/141)
  ([lbluque](https://github.com/lbluque))
* `UniformlyRandomKernel` for high temperature/random limit sampling.
`ThermalKernel` ABC class for all temperature based MC Kernels.
  [\#134](https://github.com/CederGroupHub/smol/pull/134)
  ([lbluque](https://github.com/lbluque))
* `cofe.wrangling.select` structure selection functions.
[\#133](https://github.com/CederGroupHub/smol/pull/133)
  ([lbluque](https://github.com/lbluque))
* `RegressionData` dataclass to save regression details in `ClusterExpansions`
[\#132](https://github.com/CederGroupHub/smol/pull/132)
  ([lbluque](https://github.com/lbluque))
* `rotate` method in SiteBasis class.
[\#130](https://github.com/CederGroupHub/smol/pull/130)
  ([lbluque](https://github.com/lbluque))

### Changed
* `StructureWrangler` based on pymatgen `ComputedStructureEntry`.
  [\#189](https://github.com/CederGroupHub/smol/pull/189)
  ([lbluque](https://github.com/lbluque))
* unittests for `smol.cofe` using `pytest`.
  [\#159](https://github.com/CederGroupHub/smol/pull/159)
  ([lbluque](https://github.com/lbluque))
* New `corr_from_occupancy` and `delta_corr` faster and cleaner
  implementations. And renamed `CEProcessor` to `ClusterExpansionProcessor`
  [\#156](https://github.com/CederGroupHub/smol/pull/156)
  ([lbluque](https://github.com/lbluque))
* Dropped "er" endings for `MCUsher` names. Renamed `MuSemigrandEnsemble`
  to `SemigrandEnsemble`.
  [\#154](https://github.com/CederGroupHub/smol/pull/154)
  ([lbluque](https://github.com/lbluque))
* Changed `ClusterSubspace.supercell_orbit_mappings` to only include cluster
  site indices.
  [#145](https://github.com/CederGroupHub/smol/pull/145)
([lbluque](https://github.com/lbluque))
* Enable setting cluster cutoffs for duplicate searching.
[#142](https://github.com/CederGroupHub/smol/pull/142)
([lbluque](https://github.com/lbluque))
* Methods `orbits_from_cutoffs` and `function_inds_from_cutoffs` now allow a
  dictionary as input to pick out orbits with different cluster diameter
  cutoffs.
  [\#135](https://github.com/CederGroupHub/smol/pull/135)
  ([lbluque](https://github.com/lbluque))

### Fixed
* Allow Ewald only MC.
  [\#141](https://github.com/CederGroupHub/smol/pull/169)
  ([kamronald](https://github.com/kamronald))
* Fix [141](https://github.com/CederGroupHub/smol/issues/140) corrected
  implementation of correlation function hierarchy.
  [\#141](https://github.com/CederGroupHub/smol/pull/141)
  ([lbluque](https://github.com/lbluque))
* Fix [129](https://github.com/CederGroupHub/smol/issues/129)
  saving bit_combos in `Orbit.as_dict` when pruning has been done.
  [\#130](https://github.com/CederGroupHub/smol/pull/130)
  ([qchempku2017](https://github.com/qchempku2017))
* Fix orbit generation to play nicely with changes in pymatgen
  `Structure.sites_in_sphere` return value.
  [\#125](https://github.com/CederGroupHub/smol/pull/125)
  ([lbluque](https://github.com/lbluque))
* Fix cluster searching issue
  [#104](https://github.com/CederGroupHub/smol/issues/104) when generating
  orbits from cutoffs. [#138](https://github.com/CederGroupHub/smol/pull/125)
  ([qchempku2017](https://github.com/qchempku2017))

### Removed
* `optimize_indicator` in `ClusterExpansionProcessor` and corresponding cython
   function.
  [\#156](https://github.com/CederGroupHub/smol/pull/156)
  ([lbluque](https://github.com/lbluque))
* `FuSemiGrandEnsemble` now `FugacityBias`.
  [\#154](https://github.com/CederGroupHub/smol/pull/154)
  ([lbluque](https://github.com/lbluque))
* Numerical conversion of coefficients between bases
  `ClusterExpansion.convert_coefs`
  [\#149](https://github.com/CederGroupHub/smol/pull/149)
  ([lbluque](https://github.com/lbluque))

## [alpha1.0.1](https://github.com/CederGroupHub/smol/tree/alpha1.0.1) (2021-03-03)
#### [Full Changelog](https://github.com/CederGroupHub/smol/compare/alpha1.0.0...alpha1.0.1)

### Added
* Method in `StructureWrangler` to get structure matching duplicates
  [\#122](https://github.com/CederGroupHub/smol/pull/122)
  ([lbluque](https://github.com/lbluque))
* Include tolerance when detecting duplicate correlation vectors.
  [\#121](https://github.com/CederGroupHub/smol/pull/122)
  ([lbluque](https://github.com/lbluque))
* Convenience method to get feature matrix orbit ranks.
  [\#117](https://github.com/CederGroupHub/smol/pull/117)
  ([lbluque](https://github.com/lbluque))
* bit combo hierarchy in `ClusterSubspace` for fitting hierarchy constraints.
  [\#106](https://github.com/CederGroupHub/smol/pull/106)
  ([qchempku2017](https://github.com/qchempku2017))
* data indices in `StructureWrangler` to keep track of training/test splits,
  duplicate sets, etc.
  [\#108](https://github.com/CederGroupHub/smol/pull/108)
  ([lbluque](https://github.com/lbluque))
* `ClusterSubspace.cutoffs` property to obtain tight cutoffs of included
   orbits.
   [\#108](https://github.com/CederGroupHub/smol/pull/108)
   ([lbluque](https://github.com/lbluque))
* Added properties to get orbit and ordering multiplicities of corr functions.
[\#102](https://github.com/CederGroupHub/smol/pull/102)
([lbluque](https://github.com/lbluque))
  - `function_ordering_multiplicities`, `function_total_multiplicities`
* Added helpful methods/properties to `ClusterSubspace` to get corr function
  indices based on cluster diameter cutoffs and/or cluster sizes.
[\#102](https://github.com/CederGroupHub/smol/pull/102)
([lbluque](https://github.com/lbluque))
  - `orbits_by_cutoffs`, `function_inds_by_cutoffs`, `function_inds_by_size`.

### Changed
* Allow using external term values when detecting duplicate corr vectors.
[\#124](https://github.com/CederGroupHub/smol/pull/124)
([lbluque](https://github.com/lbluque))
* Warn instead of printing when structure matching fails.
[\#124](https://github.com/CederGroupHub/smol/pull/124)
([lbluque](https://github.com/lbluque))
* filter functions in `smol.wrangling` replaced with functions returning
  indices corresponding to structures to keep. This can be used saving indices
  with `StructureWrangler.add_data_indices`.
[\#102](https://github.com/CederGroupHub/smol/pull/108)
([lbluque](https://github.com/lbluque))
* Cleanup of sites, active sites and restricted sites in `Sublattice`
[\#95](https://github.com/CederGroupHub/smol/pull/95)
  ([juliayang](https://github.com/juliayang))
* Make feature matrix optional when creating a `ClusterExpansion` construction.
[\#102](https://github.com/CederGroupHub/smol/pull/102)
([lbluque](https://github.com/lbluque))
* Renamed `ncorr_functions_per_orbit` -> `num_functions_per_orbit` in
  `ClusterSubspace` and `ClusterExpansion.convert_eci` ->
  `ClusterExpansion.convert_coefs`
[\#102](https://github.com/CederGroupHub/smol/pull/102)
([lbluque](https://github.com/lbluque))
* Changed orthonormalization of site basis to use `np.linalg.qr`.
[\#102](https://github.com/CederGroupHub/smol/pull/102)
([lbluque](https://github.com/lbluque))
* Changed cython corr functions to reduce Python interaction in loop
(~1.5x faster)
[\#102](https://github.com/CederGroupHub/smol/pull/102)
([lbluque](https://github.com/lbluque))

### Fixed
* Raise error in `StructureWrangler.append_data_items` when item properties are
  missing keys already included.
[\#117](https://github.com/CederGroupHub/smol/pull/117)
  ([lbluque](https://github.com/lbluque))
* Correctly recreate coefs in `CompositeProcessor.from_dict`
[\#116](https://github.com/CederGroupHub/smol/pull/116)
  ([lbluque](https://github.com/lbluque))
* Disallow setting chemical potentials/fugacities with duplicate string/species
  in dictionary. [\#114](https://github.com/CederGroupHub/smol/pull/114)
  ([lbluque](https://github.com/lbluque))
* Fixed loading `ClusterSubspace` with polynomial basis from dict.
[\#112](https://github.com/CederGroupHub/smol/pull/112)
  ([lbluque](https://github.com/lbluque))
* Fixed `Sublattice` serialization, saving/loading `SiteSpaces`.
[\#96](https://github.com/CederGroupHub/smol/pull/96)
  ([lbluque](https://github.com/lbluque))
* Fix json serialization when saving `ClusterSubspaces` with orthonormal basis
sets. [\#90](https://github.com/CederGroupHub/smol/pull/90)
  ([lbluque](https://github.com/lbluque))

## [alpha1.0.0](https://github.com/CederGroupHub/smol/releases/tag/alpha1.0.0) (2020-10-27)
#### [Full Changelog](https://github.com/CederGroupHub/smol/compare/alpha0.0.0...alpha1.0.0)
### Added
* Completely new `smol.moca` module. Design based generally up as follows:
  *  `Processor` classes used to compute features, properties and their local
  changes from site flips for fixed supercell sizes.
     * `ClusterExpansionProcessor` to handle cluster expansions.
     * `EwaldProcessor` to handle Ewald electrostatic energy.
     * `CompositeProcessor` to mix energy models. Currently only the ones above.
  * `Ensemble` classes to represent the corresponding statistical ensemble
  (probability space). These classes no longer run monte carlo, they only
  compute the corresponding relative Boltzman probabilities.
     * `CanonicalEnsemble` for fixed compositions.
     * `MuSemigrandEnsemble` for fixed chemical potentials.
     * `FuSemigrandEnsemble` for fixed fugacity fractions.
  * `Sublattice` class to encapsulate previous implementation using
  dictionaries.
  * `Sampler` class to run MCMC trajectories based on the given kernel using
  a specific ensemble. [\#80](https://github.com/CederGroupHub/smol/pull/80)
  ([lbluque](https://github.com/lbluque))
  * `MCKernel` classes used to implement specific MCMC algorithms.
     `Metropolis` currently only kernel implemented to run single site
     Metropolis random walk.
  * `MCUsher` classes to handle specific MCMC step proposals (i.e. single
  swaps, to preserve composition, single flips, single constrained flips,
  multisite flips, local flips, etc).
  * `SampleContainer` class to hold MCMC samples and pertinent information for
  post-processing and analysis (improvement on previous implementation using
  lists).
* `smol.moca` unit-tests all now using `pytestst` instead of `unittest`.
* `Vacancy` class, inherits from `pymatgen.DummySpecie`.
* `SiteSpace` class to encapsulate prior site space implementation using
OrderedDicts.
* `get_species` function to mimic `get_el_sp` from pymatgen but correctly
handle `Vacancy`.
* MCMC sample streaming functionality using hdf5 files.
[\#84](https://github.com/CederGroupHub/smol/pull/84)
([lbluque](https://github.com/lbluque))
* Initial Sphinx code documentation, currently hosted [here](http://amox.lbl.gov/smol).
* `StructureWrangler` warning when adding structures with duplicate correlation
vectors. [\#85](https://github.com/CederGroupHub/smol/pull/85)
([lbluque](https://github.com/lbluque))

### Changed
* `Ensemble` classes are temperature agnostic. Temperature is set when sampling.
[\#83](https://github.com/CederGroupHub/smol/pull/83)
([lbluque](https://github.com/lbluque))
* Refactored `smol.cofe.configspace` -> `smol.cofe.space`
* A few method name changes in `ClusterSubspace` to be more precise and
appropriate. Most notably `from_radii` classmethod now `from_cutoffs` (since
the distances used, max distance between 2 pts, are more like a diameter rather
than a radius.)
* filtering functions no longer methods in `StructureWrangler`, now defined
as functions in `cofe.wrangling.filter`.
[\#85](https://github.com/CederGroupHub/smol/pull/85)
([lbluque](https://github.com/lbluque))
* Species in site spaces and occupancy strings are now pymatgen `Specie` or
inherited classes instead of string names. (This is to allow keeping additional
properties for species, such as oxidation state, magnetization, the sky is the
limit.)
* Single `StandardBasis` site basis class that is constructed using a basis
function iterator for specific basis sets.
* Example notebooks updated accordingly.

### Removed
* `smol.learn` and all regression estimators have been removed.

### Fixed
* Proper calculation of ECIs in `ClusterExpansion` using both random
crystallographic symmetry multiplicity and function decoration multiplicity.
(credits to [qchempku2017](https://github.com/qchempku2017) for pointing this
out.)
* Fixed MSONable serialization of cluster subspaces with orthonormal basis sets
by making `StandardBasis` MSONable and saving corresponding arrays.
[\#90](https://github.com/CederGroupHub/smol/pull/90)


## [alpha0.0.0](https://github.com/CederGroupHub/smol/tree/alpha0.0.0) (2020-10-8)
Initial relatively *stable* version of the code.
