
.. title:: smol documentation

.. figure:: ../images/logo.png
   :alt: server
   :align: left
   :width: 600px

Statistical Mechanics on Lattices
=================================

*Lighthweight but caffeinated Python implementations of computational methods
for statistical mechanical calculations of configurational states for
crystalline material systems.*

.. image:: https://github.com/CederGroupHub/smol/actions/workflows/test.yml/badge.svg
      :alt: Test Status
      :target: https://github.com/CederGroupHub/smol/actions/workflows/test.yml

-------------------------------------------------------------------------------

**smol** is a minimal implementation of computational methods to calculate
statistical mechanical and thermodynamic properties of crystalline
material systems based on the *cluster expansion* method from alloy theory and
related methods. Although **smol** is intentionally lightweight---in terms of
dependencies and built-in functionality---it has a modular design that closely
follows underlying mathematical formalism and provides useful abstractions to
easily extend existing methods or implement and test new ones. Finally,
although conceived mainly for method development, **smol** can (and is being)
used in production for materials science reasearch applications.



Installation
============
1.  Clone the repository.
2.  Go to the top level directory of the cloned repo and type::

        pip install .

Basic Usage
===========
**smol** is designed to be simple and intuitive to use. Here is the most
basic example of creating a Cluster Expansion for a binary alloy and
subsequently using it to run Monte Carlo sampling::

    from sklearn.linear_model import LinearRegression
    from monty.serialization import loadfn
    from pymatgen.core.structure import Structure
    from pymatgen.transformations.standard_transformations import \
        OrderDisorderedStructureTransformation
    from smol.cofe import ClusterSubspace, StructureWrangler, \
        ClusterExpansion, RegressionData
    from smol.moca import CanonicalEnsemble, Sampler
    from smol.io import save_work

    # load some dataset with structures and corresponding energies
    data = loadfn("path_to_file.json")

    # create a disordered primitice structure
    species = {"Au": 0.5, "Cu": 0.5}
    prim = Structure.from_spacegroup(
                "Fm-3m", Lattice.cubic(3.6), [species], [[0, 0, 0]])

    # create a clustersubspace
    cutoffs = {2: 6, 3: 5, 4: 4}
    subspace = ClusterSubspace.from_cutoffs(prim, cutoffs=cutoffs)

    # prepare fit data
    wrangler = StructureWrangler(subspace)
    for structure, energy in data:
        wrangler.add_data(structure, properties={"energy": energy})

    # fit the expansion with OLS
    reg = LinearRegression(fit_intercept=False)
    reg.fit(wrangler.feature_matrix,
            wrangler.get_property_vector("energy"))

    # save details of the regression model used
    reg_data = RegressionData.from_sklearn(
        estimator=reg,
        feature_matrix=wrangler.feature_matrix,
        property_vector=wrangler.get_property_vector('energy'))

    expansion = ClusterExpansion(
        subspace, coefficients=reg.coef_, regression_data=reg_data)

    # define a supercell and ensemble for MC sampling
    sc_matrix = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
    ensemble = CanonicalEnsemble.from_cluster_expansion(
        expansion, supercell_matrix=sc_matrix)

    sampler = Sampler.from_ensemble(
        ensemble, temperature=500)

    # Get an initial ordered structure for 5x5x5 supercell using pymatge
    transformation = OrderDisorderedStructureTransformation()
    structure = expansion.cluster_subspace.structure.copy()
    structure.make_supercell(sc_matrix)
    structure = transformation.apply_transformation(structure)

    # Create initial occupancy and run MCMC
    init_occu = ensemble.processor.occupancy_from_structure(structure)
    sampler.run(1000000, initial_occupancy=init_occu)

    save_work(
        "CuAu_ce_mc.json", wrangler, expansion, ensemble, sampler.samples)


API Documentation
=================
See the :doc:`api` documentation page for in depth reference to core classes
and functions.

Detailed Examples
=================
You can find more in-depth and advanced usage examples in the
:doc:`examples` page.

==============
Recent Changes
==============
You can find updates and the most recent changes in the
`Changelog <https://github.com/CederGroupHub/smol/blob/master/CHANGES.md>`_.

=====================
Help, Issues, Support
=====================
To get immediate help ask in the #cluster-expansion slack channel. For more
detailed issues, bug reports and requests please submit a
`Github issue <https://github.com/CederGroupHub/smol/issues>`_.

============
Contributing
============
To contribute bug fixes or new code please refer to the contributing
`guidelines <https://github.com/CederGroupHub/smol/blob/master/CONTRIBUTING.md>`_.

