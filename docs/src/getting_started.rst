.. _geting-started :

===============
Getting Started
===============


Installation
============
**smol** is purposely light on dependencies which should make the installation
process headache free. Using ``pip``::

        pip install smol

Although **smol** is not tested on Windows platforms, it should still run on Windows
since there aren't any platform specific dependencies. The only known installation issue
is building *pymatgen* dependencies. If simply running ``pip install smol`` fails, try
installing *pymatgen* with conda first::

        conda install -c conda-forge pymatgen
        pip install smol

Basic Usage
===========

**smol** is designed to be simple and intuitive to use. Here is the most
basic example of creating a Cluster Expansion for a binary alloy and
subsequently using it to run Monte Carlo sampling.

Creating a cluster subspace
---------------------------
Create a cluster subspace for a AuCu binary FCC alloy to define the cluster
expansion terms and compute the corresponding correlation functions.

Start by creating a disordered primitive structure.

.. code-block:: python

    from pymatgen.core.structure import Structure, Lattice

    species = {"Au": 0.5, "Cu": 0.5}
    prim = Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.6), [species], [[0, 0, 0]])

Now create a cluster subspace for that structure including pair, triplet and
quadruplet clusters up to given cluster diameter cutoffs.

.. code-block:: python

    from smol.cofe import ClusterSubspace

    cutoffs = {2: 6, 3: 5, 4: 4}
    subspace = ClusterSubspace.from_cutoffs(prim, cutoffs=cutoffs)

Preparing training data
-----------------------

Load and use data computed for a training set of ordered structures to
generate the necessary fitting data (formation energy and correlation vector
for each training structure). Training data is added as instances of
`ComputedStructureEntry <https://pymatgen.org/pymatgen.entries.computed_entries.html?highlight=computedstructureentry#pymatgen.entries.computed_entries.ComputedStructureEntry>`_

.. code-block:: python

    from monty.serialization import loadfn
    from smol.cofe import StructureWrangler

    entries = loadfn("path_to_file.json")
    wrangler = StructureWrangler(subspace)
    for entry in entries:
        wrangler.add_entry(entry)

Fitting and creating a cluster expansion
----------------------------------------

Using the generated feature matrix and property vector fit a cluster expansion.
In this case we use simple linear regression, although for most cases this will
not be appropriate and a regularized regression model will yield a much better
fit.

.. code-block:: python

    from sklearn.linear_model import LinearRegression

    reg = LinearRegression(fit_intercept=False)
    reg.fit(wrangler.feature_matrix, wrangler.get_property_vector("energy"))

Finally, create a cluster expansion for prediction of new structures and
eventual Monte Carlo sampling. We recommend saving the details used to fit the
expansion for future reproducibility (although this is not strictly necessary).

.. code-block:: python

    from smol.cofe import ClusterExpansion, RegressionData

    reg_data = RegressionData.from_sklearn(
        estimator=reg,
        feature_matrix=wrangler.feature_matrix,
        property_vector=wrangler.get_property_vector("energy"),
    )
    expansion = ClusterExpansion(subspace, coefficients=reg.coef_, regression_data=reg_data)

Creating an ensemble for Monte Carlo Sampling
---------------------------------------------

Creating an ensemble only requires the cluster expansion and a supercell matrix
to define the sampling domain.

.. code-block:: python

    from smol.moca import Ensemble

    sc_matrix = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
    ensemble = Ensemble.from_cluster_expansion(expansion, supercell_matrix=sc_matrix)

Running Monte Carlo sampling
----------------------------
To generate MC samples for the ensemble, we need to create a sampler
object.

.. code-block:: python

    from smol.moca import Sampler

    sampler = Sampler.from_ensemble(ensemble, temperature=1000)

In order to begin an MC simulation, an initial configuration must be provided.
In this case we use pymatgen's functionality to provide an ordered structure
given a disordered one.

.. code-block:: python

    from pymatgen.transformations.standard_transformations import (
        OrderDisorderedStructureTransformation,
    )

    transformation = OrderDisorderedStructureTransformation()
    structure = expansion.cluster_subspace.structure.copy()
    structure.make_supercell(sc_matrix)
    structure = transformation.apply_transformation(structure)

Finally, the ordered structure can be used to generate an initial configuration
to run MC sampling.

.. code-block:: python

    init_occu = ensemble.processor.occupancy_from_structure(structure)
    sampler.run(100000, initial_occupancy=init_occu)

Saving the generated objects and data
-------------------------------------
To save the generated objects for the previous workflow we can simply use the
provided convenience io functionality. However, all main classes are
serializable just as pymatgen and so can be saved as json dictionaries or
using the `monty <https://guide.materialsvirtuallab.org/monty//>`_ python
package.

.. code-block:: python

    save_work("CuAu_ce_mc.json", wrangler, expansion, ensemble, sampler.samples)


.. code-links:: python
.. code-links:: clear


Example Notebooks
=================

For more detailed examples on how to use **smol** have a look at the following
Jupyter notebooks.

You can run the notebooks interactively on Binder.

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/CederGroupHub/smol/HEAD?labpath=docs%2Fsrc%2Fnotebooks%2Findex.ipynb

Basic Examples
--------------

- `Creating a basic cluster expansion`_
- `Creating a cluster expansion with electrostatics`_
- `Visualizing clusters`_
- `Running canonical Monte Carlo`_
- `Running semigrand canonical Monte Carlo`_
- `Running charge neutral semigrand canonical Monte Carlo`_
- `Setting number of threads for OpenMP parallelization`_


.. _Creating a basic cluster expansion: notebooks/creating-a-ce.ipynb

.. _Creating a cluster expansion with electrostatics: notebooks/creating-a-ce-w-electrostatics.ipynb

.. _Visualizing clusters: notebooks/cluster-visualization.ipynb

.. _Running Canonical Monte Carlo: notebooks/running-canonical-mc.ipynb

.. _Running Semigrand Canonical Monte Carlo: notebooks/running-semigrand-mc.ipynb

.. _Running Charge Neutral Semigrand Canonical Monte Carlo: notebooks/running-charge-balanced-gcmc.ipynb

.. _Setting number of threads for OpenMP parallelization: notebooks/openmp-parallelism.ipynb


Advanced Examples
-----------------

- `Preparing cluster expansion training data`_
- `Centering training data in stage-wise fit with electrostatics`_
- `Adding structures to a StructureWrangler in parallel`_
- `Simulated annealing with point electrostatics`_
- `Wang-Landau sampling of an FCC Ising model`_
- `Generating special quasirandom structures`_
- `Li-Mn-O DRX cluster expansion and sampling`_

.. _Preparing cluster expansion training data: notebooks/training-data-preparation.ipynb

.. _Centering training data in stage-wise fit with electrostatics: notebooks/ce-fit-w-centering.ipynb

.. _Adding structures to a StructureWrangler in parallel: notebooks/adding-structures-in-parallel.ipynb

.. _Simulated annealing with point electrostatics: notebooks/running-ewald-sim_anneal.ipynb

.. _Wang-Landau sampling of an FCC Ising model: notebooks/wang-landau-ising.ipynb

.. _Generating special quasirandom structures: notebooks/generating-sqs.ipynb

.. _Li-Mn-O DRX cluster expansion and sampling: notebooks/lmo-drx-ce-mc.ipynb

More to come...
