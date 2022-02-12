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
