==========
User Guide
==========

**smol** is organized into two main submodules:

- :ref:`smol.cofe ug` (Cluster Orbit Fourier Expansion) includes classes and
  functions to define, train, and test cluster expansions.
- :ref:`smol.moca ug` (Monte Carlo) includes classes and functions to run
  Markov Chain Monte Carlo (MCMC) sampling based on a cluster expansion
  Hamiltonian (and a few other Hamiltonian models).

Below is the documentation for the core classes in each submodule.
You can also refer to the :ref:`api ref` for documentation of all classes
and functions in the package.

.. _smol.cofe ug:

=========
smol.cofe
=========

This module includes the necessary classes to define, train, and test cluster
expansions. A cluster expansion is essentially a way to fit a function of
configurational degrees of freedom using a specific set of basis functions that
allow a sparse representation of that function (which resides in a high
dimensional function space). For a more thorough treatment of the formalism of
cluster expansions refer to this document or any of following references
[`Sanchez et al., 1993 <https://doi.org/10.1103/PhysRevB.48.14013>`_,
`Ceder et al., 1995 <https://doi.org/10.1103/PhysRevB.51.11257>`_,
`van de Walle et al., 2009 <https://doi.org/10.1016/j.calphad.2008.12.005>`_].

The core classes are:

- :ref:`cluster subspace ug`
- :ref:`structure wrangler ug`
- :ref:`cluster expansion ug`

.. _cluster subspace ug:

Cluster subspace
----------------
:class:`ClusterSubspace` contains the finite set of orbits and orbit basis
functions to be included in the cluster expansion.
In general, a cluster expansion is created by first generating a
:class:`ClusterSubspace`, which uses a provided primitive cell of the
pymatgen `Structure <https://pymatgen.org/pymatgen.core.structure.html>`_
class to build the orbits of the cluster expansion. Because orbits generally
decrease in importance with length, it is recommended to use the convenience
method :meth:`from_cutoffs` to specify the cutoffs of different size
orbits (pairs, triplets, quadruplets, etc.) In addition to specifying the
type of site basis functions and their orthonormality,
:class:`ClusterSubspace` also has capabilities for matching fitting structures
and determining site mappings to compute correlation vectors. Full
documentation of the class is available here: :ref:`cluster subspace`.

.. _structure wrangler ug:

Structure wrangler
------------------
:class:`StructureWrangler` handles input data structures and properties
to fit to the cluster expansion.
Once a set of structures and their relevant properties (for example, their
volume or energies) have been obtained (e.g., through first-principles
calculations), :class:`StructureWrangler` can be used to process this data.
Specifically, based on a given :class:`ClusterSubspace`,
:class:`StructureWrangler` can to compute correlation vectors and convert
the input structure data into a feature matrix for fitting to the property
vector. Additional methods are available to help process the input data,
including methods for checking, preparing, and filtering the data. Full
documentation of the class is available here: :ref:`structure wrangler`.

.. _cluster expansion ug:

Cluster expansion
-----------------
:class:`ClusterExpansion` contains the fitted coefficents of the cluster
expansion for predicting CE properties of new structures.
Based on the feature matrix from the :class:`StructureWrangler`, one can fit
fit the data to the properties using any fitting method they like (e.g.,
linear regression, regularized regression, etc). :code:`smol.cofe`
contains wrapper class :class:`RegressionData` for regression methods from
`sklearn <https://scikit-learn.org/stable/>`_. The fitted coefficients and
:class:`ClusterSubspace` objects are then given to :class:`ClusterExpansion`.
The :class:`ClusterExpansion` object can be used to predict the properties
of new structures but more importantly can be given to the :ref:`smol.moca ug`
classes for MC sampling. Full documentation of the class is available here:
:ref:`cluster expansion`.


.. _smol.moca ug:

=========
smol.moca
=========

This module includes classes and functions to run Markov Chain Monte Carlo
sampling of statistical mechanical ensembles represented by a cluster expansion
Hamiltonian (there is also support to run MCMC with simple pair interaction
models, such as Ewald electrostatic interactions). MCMC sampling is done for a
specific supercell size. In theory the larger the supercell the better the
results, however in practice there are many other nuances for picking the right
supercell size that are beyond the scope of this documentation. Refer to the
following references for appropriate expositions of the method [].

The core classes are:

- :ref:`ensembles ug`

  - :class:`CanonicalEnsemle`
  - :class:`SemiGrandEnsemble`

- :ref:`processors ug`

  - :class:`CEProcessor`
  - :class:`EwaldProcessor`
  - :class:`CompositeProcessor`

- :ref:`sampler ug`
- :ref:`samplecontainer ug`

.. _ensembles ug:

Ensembles
---------
:class:`Ensemble` classes represent the specific statistical mechanics ensemble
by defining the relevant thermodynamic boundary conditions in order to compute
the appropriate ensemble probability ratios. For example,
:class:`CanonicalEnsemble` is used for systems at constant temperature and
constant composition, while :class:`SemiGrandEnsemble` is used for systems at
constant temperature and constant chemical potential. Ensembles also hold
information of the underlying set of :class:`Sublattice` for the configuration
space to be sampled. Note that as implemented, an ensemble applies to any
temperature, but the specific temperature to generate samples at is set in a
:class:`Sampler`. Full documentation of the class and its subclasses are
available here: :ref:`ensembles`.

.. _processors ug:

Processors
----------
A :class:`Processor` is used to optimally compute correlation vectors, energy,
and differences in these from variations in site occupancies. Processors
compute values only for a specific supercell specified by a given supercell
matrix.

Users will rarely need to directly instantiate a processor, and it is recommended
to simply create an ensemble using the :meth:`from_cluster_expansion` which
will automatically instantiate the appropriate processor. Then, accessing the
processor can be done simply by the corresponding attribute (i.e.
:code:`ensemble.processor`). Many methods and attributes of a processor are
very useful for setting up and analysing MCMC sampling runs. Full
documentation of the class and its subclasses available here:
:ref:`processors`.

.. _sampler ug:

Sampler
-------
A :class:`Sampler` takes care of running MCMC sampling for a given ensemble.
The easiest way to create a sampler (which suffices for most use cases) is to
use the :meth:`from_ensemble` class method. For more advanced use cases and
elaborate MCMC sampling more knowledge of the underlying classes (especially
:class:`Metropolis` which applies the `Metropolis-Hastings algorithm
<https://doi.org/10.1093/biomet/57.1.97>`_ and
:class:`MCUsher` which proposes relevant flips) is necessary. Full
documentation of the class is available here: :ref:`sampler`.

.. _samplecontainer ug:

SampleContainer
---------------
A :class:`SampleContainer` stores data from Monte Carlo sampling simulations,
especially the occupancies and feature vectors. It also includes some minimal
methods and properties useful to begin analysing the raw samples, including
methods to obtain the mean/variance/minimum of energies, enthalpies, and
composition. Full documentation of the class is available here:
:ref:`sample container`.