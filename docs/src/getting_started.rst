===============
Getting Started
===============


Installation
============
**smol** is purposedly light on dependencies which should make the installation
process headache free.

        pip install statmech-on-lattices

(unfortunately PyPi hates fun so we could use "smol".)

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

.. nbplot::

    >>> from pymatgen.core.structure import Structure
    >>> species = {"Au": 0.5, "Cu": 0.5}
    >>> prim = Structure.from_spacegroup(
            "Fm-3m", Lattice.cubic(3.6), [species], [[0, 0, 0]])

Now create a cluster subspace for that structure including pair, triplet and
quadruplet clusters up to given cluster diameter cutoffs.

.. nbplot::

    >>> from smol.cofe import ClusterSubspace
    >>> cutoffs = {2: 6, 3: 5, 4: 4}
    >>> subspace = ClusterSubspace.from_cutoffs(prim, cutoffs=cutoffs)

Preparing training data
-----------------------

Load and use data computed for a training set of ordered structures to
generate the necessary fitting data---formation energy and correlation vector
for each training point.

.. nbplot::

    >>> from monty.serialization import loadfn
    >>> from smol.cofe import StructureWrangler
    >>> data = loadfn("path_to_file.json")
    >>> wrangler = StructureWrangler(subspace)
    >>> for structure, energy in data:
            wrangler.add_data(structure, properties={"energy": energy})

Fitting and creating a cluster expansion
----------------------------------------

Using the generated feature matrix and property vector fit a cluster expansion.
In this case we use simple linear regression, although for most cases this will
not be appropriate and a regularized regression model will yield a much better
fit.

.. nbplot::

    >>> from sklearn.linear_model import LinearRegression
    >>> reg = LinearRegression(fit_intercept=False)
    >>> reg.fit(wrangler.feature_matrix,
                wrangler.get_property_vector("energy"))

Finally, create a cluster expansion for prediction of new structures and
eventual Monte Carlo sampling. We recommed saving the details used to fit the
expansion for future reproducibility (although this is not strictly necessary).

.. nbplot::

    >>> from smol.cofe import ClusterExpansion, RegressionData
    >>> reg_data = RegressionData.from_sklearn(
            estimator=reg,
            feature_matrix=wrangler.feature_matrix,
            property_vector=wrangler.get_property_vector('energy'))
    >>> expansion = ClusterExpansion(
            subspace, coefficients=reg.coef_, regression_data=reg_data)

Creating an ensemble for Monte Carlo Sampling
---------------------------------------------

Creating an ensemble only requires the cluster expansion and a supercell matrix
to define the sampling domain.

.. nbplot::

    >>> from smol.moca import CanonicalEnsemble
    >>> sc_matrix = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
    >>> ensemble = CanonicalEnsemble.from_cluster_expansion(
            expansion, supercell_matrix=sc_matrix)

Running Monte Carlo sampling
----------------------------
To generate MC samples for the ensemble, we need to create a sampler
object.

.. nbplot::

    >>> from smol.moca import Sampler
    >>> sampler = Sampler.from_ensemble(
            ensemble, temperature=500)

In order to begin an MC simulation, an initial configuration must be provided.
In this case we use pymatgen's functionality to provide an ordered structure
given a disordered one.

.. nbplot::

    >>> from pymatgen.transformations.standard_transformations import \
            OrderDisorderedStructureTransformation
    >>> transformation = OrderDisorderedStructureTransformation()
    >>> structure = expansion.cluster_subspace.structure.copy()
    >>> structure.make_supercell(sc_matrix)
    >>> structure = transformation.apply_transformation(structure)

Finally, the ordered structure can be used to generate an initial configuration
to run MC sampling interations.

.. nbplot::

    >>> init_occu = ensemble.processor.occupancy_from_structure(structure)
    >>> sampler.run(1000000, initial_occupancy=init_occu)

Saving the generated objects and data
-------------------------------------
To save the generated objects for the previous workflow we can simply use the
provided convenience io functionaltiy. However, all main classes are
serializable just as pymatgen and so can be saved as json dictionaries or
using the `monty <https://guide.materialsvirtuallab.org/monty//>`_ python
package.

.. nbplot::

    >>> save_work(
        "CuAu_ce_mc.json", wrangler, expansion, ensemble, sampler.samples)


.. code-links:: python
.. code-links:: clear


Example Notebooks
=================
For more detailed examples on how to use **smol** have a look at the following
Jupyter notebooks.

Basic Examples
--------------

- `Creating a basic cluster expansion`_
- `Creating a cluster expansion with electrostatics`_
- `Running Canonical Monte Carlo`_
- `Running Semi-Grand Canonical Monte Carlo`_
- `Preparing cluster expansion training data`_
- `Adding structures to a StructureWrangler in parallel`_

.. _Creating a basic cluster expansion: notebooks/1-creating-a-ce.ipynb

.. _Creating a cluster expansion with electrostatics: notebooks/1-1-creating-a-ce-w-electrostatics.ipynb

.. _Running Canonical Monte Carlo: notebooks/2-running-canonical-mc.ipynb

.. _Running Semi-Grand Canonical Monte Carlo: notebooks/2-1-running-semigrand-mc.ipynb

.. _Preparing cluster expansion training data: notebooks/3-training-data-preparation.ipynb

.. _Adding structures to a StructureWrangler in parallel: notebooks/4-adding-structures-in-parallel.ipynb

Advanced Examples
-----------------
Soon to come...
