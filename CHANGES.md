# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
Use this section to keep track of changes in the works.

### Added
* data indices in `StructureWrangler` to keep track of training/test splits,
  duplicate sets, etc.
* `ClusterSubspace.cutoffs` property to obtain tight cutoffs of included
   orbits.
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
* Fixed `Sublattice` serialization, saving/loading `SiteSpaces`.
[\#96](https://github.com/CederGroupHub/smol/pull/96)
  ([lbluque](https://github.com/lbluque))
* Fix json serialization when saving `ClusterSubspaces` with orthonormal basis
sets. [\#90](https://github.com/CederGroupHub/smol/pull/90)
  ([lbluque](https://github.com/lbluque))

## [v1.0.0]() (2020-10-27)
#### [Full Changelog](https://github.com/CederGroupHub/smol/compare/v0.0.0...v1.0.0)
### Added
* Completely new `smol.moca` module. Design based generally up as follows:
  *  `Processor` classes used to compute features, properties and their local
  changes from site flips for fixed supercell sizes.
     * `CEProcessor` to handle cluster expansions.
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
  * `MCMCKernel` classes used to implement specific MCMC algorithms.
     `Metropolis` currently only kernel implemented to run single site
     Metropolis random walk.
  * `MCMCUsher` classes to handle specific MCMC step proposals (i.e. single 
  swaps, to preseve composition, single flips, single constrained flips,
  multisite flips, local flips, etc).
  * `SampleContainer` class to hold MCMC samples and pertinent information for
  post-processing and analysis (improvement on previous implementation using
  lists).
* `smol.moca` unit-tests all now using `pytestst` instead of `unittest`.
* `Vacancy` class, inherits from `pymatgen.DummySpecie`.
* `SiteSpace` class to encapsulate prior site space implementation using
OrderedDicts.
* `get_species` function to mimick `get_el_sp` from pymatgen but correctly
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
* Single `SiteBasis` site basis class that is constructed using a basis
function iterator for specific basis sets.
* Example notebooks updated accordingly.

### Deprecated

### Removed
* `smol.learn` and all regression estimators have been removed.

### Fixed
* Proper calculation of ECIs in `ClusterExpansion` using both random
crystallographic symmetry multiplicity and function decoration multiplicity.
(credits to [qchempku2017](https://github.com/qchempku2017) for pointing this
out.)
* Fixed MSONable serialization of cluster subspaces with orthonormal basis sets
by making `SiteBasis` MSONable and saving corresponding arrays.
[\#90](https://github.com/CederGroupHub/smol/pull/90)


## [v0.0.0](https://github.com/CederGroupHub/smol/tree/v0.0.0) (2020-10-8)
Initial relatively *stable* version of the code.
