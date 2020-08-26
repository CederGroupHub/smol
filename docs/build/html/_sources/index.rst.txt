
.. title:: smol documentation

.. figure:: ../images/logo.png
   :alt: server
   :align: left
   :width: 600px

Statistical Mechanics on Lattices
=================================

Lighthweight but caffeinated Python implementations of computational methods
for statistical mechanical calculations of configurational states for
crystalline material systems.

.. image:: https://img.shields.io/circleci/build/gh/CederGroupHub/smol/master?logo=circleci&style=for-the-badge&token=96d0d7a959e1e12044ff45daa43218ae7fa4303e
.. image:: https://img.shields.io/codacy/coverage/4b527a2fd9ad40f59195f1f8dc1ac542?style=for-the-badge
.. image:: https://img.shields.io/codacy/grade/4b527a2fd9ad40f59195f1f8dc1ac542?style=for-the-badge

Installation
============
1.  Clone the repository.
2.  Go to the top level directory of the cloned repo and type::

        pip install .

Basic Usage
===========
**smol** is quite easy to use. Here is the most basic example of creating
a Cluster Expansion for a binary alloy and doing Monte Carlo sampling::

    from sklearn.linear_model import LinearRegression
    from monty.serialization import loadfn
    from pymatgen import Structure
    from smol.cofe import ClusterSubspace, StructureWrangler, ClusterExpansion
    from smol.moca import CanonicalEnsemble, Sampler
    from smol.io import save_work

    # load some dataset with fit structures and corresponding energies
    data = loadfn("path_to_file.json")

    species = {"Au": 0.5, "Cu": 0.5}
    prim = Structure.from_spacegroup("Fm-3m",
                                     Lattice.cubic(3.6),
                                     [species], [[0, 0, 0]])
    cutoffs = {2: 6, 3: 5, 4: 4}
    subspace = ClusterSubspace.from_radii(prim, radii=cutoffs)

    wrangler = StructureWrangler(subspace)
    for structure, energy in data:
        wrangler.add_data(structure, properties={"energy": energy})

    reg = LinearRegression(fit_intercept=False)
    reg.fit(wrangler.feature_matrix, wrangler.get_property_vector("energy"))

    expansion = ClusterExpansion(subspace, coefficients=reg.coef_,
                                 feature_matrix=wrangler.feature_matrix)

    sc_matrix = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
    ensemble.from_cluster_expansion(expansion, supercell_matrix=sc_matrix,
                                    temperature=500)

    sampler = Sampler.from_ensemble(ensemble)

    # Get some initial ordered structure for 5x5x5 supercell
    init_occu = ensemble.processor.occupancy_from_structure(structure)
    sampler.run(1000000, initial_occupancy=init_occu)

    save_work("CuAu_ce_mc.json", wrangler, expansion, ensemble,
              sampler.samples)


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
You can find the most recent chagnes in the :doc:`changelog`.

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

