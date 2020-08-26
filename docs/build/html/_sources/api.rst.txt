===
API
===

**smol** is organized into two main submodules:

- :ref:`smol.cofe` (Cluster Orbit Fourier Expansion) includes functionality
  to define, train and test cluster expansions.
- :ref:`smol.moca` (Monte Carlo) includes functionality to run MCMC sampling
  based on a cluster expansion hamiltonian (and a few other hamiltonian
  models).

Here is the documentation for the core classes in each submodule. You can also
refer to :ref:`fulldocs` for the full documentation.

.. _smol.cofe:

=========
smol.cofe
=========

This module defines the necessary classes to define, train and test cluster
expansions. A cluster expansion is essentially a way to fit a function of
configurational degrees of freedom using a specific set of basis functions that
allow a sparse representation of a function which resides in a high
dimensional function space. For a more thorough treatment of the formalism of
cluster expansions refer to this document or any of following references [].

The core classes are:

- :ref:`clustersubspace`
- :ref:`structurewrangler`
- :ref:`clusterexpansion`


.. _clustersubspace:

ClusterSubspace
---------------

.. autoclass:: smol.cofe.ClusterSubspace
   :members:
   :undoc-members:
   :show-inheritance:

.. _structurewrangler:

StructureWrangler
-----------------

.. autoclass:: smol.cofe.StructureWrangler
   :members:
   :undoc-members:
   :show-inheritance:


.. _clusterexpansion:

ClusterExpansion
----------------

.. autoclass:: smol.cofe.ClusterExpansion
   :members:
   :undoc-members:
   :show-inheritance:

.. _smol.moca:

=========
smol.moca
=========

This module defines classes to run Markov Chain Monte Carlo sampling of
statistical mechanical ensembles represented by a cluster expansion Hamiltonian
(there is also support to run MCMC with simple pair interaction models, such as
Ewald electrostatic interactions). MCMC sampling is done for a specific
supercell size. In theory the larger the supercell the better the results,
however in practice there are many other nuances for picking the right
supercell size that are beyond the scope of this documentation.

The core classes are:

- :ref:`ensembles`

  - :class:`CanonicalEnsemle`
  - :class:`MuSemiGrandEnsemble`
  - :class:`FuSemiGrandEnsemble`

- :ref:`processors`

  - :class:`CEProcessor`
  - :class:`EwaldProcessor`
  - :class:`CompositeProcessor`

- :ref:`sampler`
- :ref:`samplecontainer`

.. _ensembles:

Ensembles
---------
:class:`Ensemble` classes represent the specific statistical mechanics ensemble
by defining the relevant thermodynamic boundary conditions in order to compute
the appropriate ensemble probability ratios. Ensembles also hold information
of the underlying set of :class:`Sublattice` for the domain to be sampled.

.. autoclass:: smol.moca.ensemble.base.Ensemble
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: smol.moca.CanonicalEnsemble
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: smol.moca.ensemble.semigrand.BaseSemiGrandEnsemble
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: smol.moca.MuSemiGrandEnsemble
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: smol.moca.FuSemiGrandEnsemble
   :members:
   :undoc-members:
   :show-inheritance:


.. _processors:

Processors
----------
A class:`Processor` is used to optimally compute correlation vectors, energy,
and differences in these from variations in site occupancies. Processors
compute values only for a specific supercell specified by a supercell matrix.

A user will rarely need to instantiate a processor, it is recommended to simply
create an ensemble using the :meth:`from_cluster_expansion` which will
automatically instantiate the appropriate processor. However many methods of a
processor are very useful for seting up and analysing MCMC sampling runs.


.. autoclass:: smol.moca.processor.base.Processor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: smol.moca.CEProcessor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: smol.moca.EwaldProcessor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: smol.moca.CompositeProcessor
   :members:
   :undoc-members:
   :show-inheritance:

.. _sampler:

Sampler
-------
A class:`Sampler` takes care of running MCMC sampling for a given ensemble. The
easiest method to create a sampler (which suffices for most use cases) is to
use the :meth:`from_ensemble` class method. For more advanced use cases and
elaborate MCMC sampling more knowledge of the underlying classes is necessary.

.. autoclass:: smol.moca.Sampler
   :members:
   :undoc-members:
   :show-inheritance:

.. _samplecontainer:

SampleContainer
---------------
A :class:`SampleContainer` holds samples and sampling information from an MCMC
sampling run. It is useful to obtain the raw data and some minimal empirical
properties of the underlying distribution in order to carry out further
analysis of the MCMC sampling results.

.. autoclass:: smol.moca.SampleContainer
   :members:
   :undoc-members:
   :show-inheritance:


.. _fulldocs:

Full API Documentation
----------------------
Here is the autogenerated documentation for all of the **smol** source code.
If you find typos, grammar mistakes or confusing sentences please let
*lbluque* know or make a quick PR fixing the docstrings.

* :ref:`modindex`
* :ref:`genindex`
* :ref:`search`