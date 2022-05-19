# ## Getting Started
#
# ### Installation
#
# **smol** is purposedly light on dependencies which should make the installation
# process headache free. Using `pip`:
#
# ```
# pip install statmech-on-lattices
# ```
#
# (unfortunately someone is name-squatting so we can’t use “smol” for now.)
#
# ### Basic Usage
#
# **smol** is designed to be simple and intuitive to use. Here is the most
# basic example of creating a Cluster Expansion for a binary alloy and
# subsequently using it to run Monte Carlo sampling.
#
# #### Creating a cluster subspace
#
# Create a cluster subspace for a AuCu binary FCC alloy to define the cluster
# expansion terms and compute the corresponding correlation functions.
#
# Start by creating a disordered primitive structure.

from pymatgen.core.structure import Structure, Lattice
species = {"Au": 0.5, "Cu": 0.5}
prim = Structure.from_spacegroup(

# Now create a cluster subspace for that structure including pair, triplet and
# quadruplet clusters up to given cluster diameter cutoffs.

from smol.cofe import ClusterSubspace
cutoffs = {2: 6, 3: 5, 4: 4}
subspace = ClusterSubspace.from_cutoffs(prim, cutoffs=cutoffs)

# # Preparing training data
#
# Load and use data computed for a training set of ordered structures to
# generate the necessary fitting data (formation energy and correlation vector
# for each training structure). Training data is added as instances of
# [ComputedStructureEntry](https://pymatgen.org/pymatgen.entries.computed_entries.html?highlight=computedstructureentry#pymatgen.entries.computed_entries.ComputedStructureEntry)

from monty.serialization import loadfn
from smol.cofe import StructureWrangler
entries = loadfn("path_to_file.json")
wrangler = StructureWrangler(subspace)
for entry in entries:

# # Fitting and creating a cluster expansion
#
# Using the generated feature matrix and property vector fit a cluster expansion.
# In this case we use simple linear regression, although for most cases this will
# not be appropriate and a regularized regression model will yield a much better
# fit.

from sklearn.linear_model import LinearRegression
reg = LinearRegression(fit_intercept=False)
reg.fit(wrangler.feature_matrix,

# Finally, create a cluster expansion for prediction of new structures and
# eventual Monte Carlo sampling. We recommed saving the details used to fit the
# expansion for future reproducibility (although this is not strictly necessary).

from smol.cofe import ClusterExpansion, RegressionData
reg_data = RegressionData.from_sklearn(
expansion = ClusterExpansion(

# # Creating an ensemble for Monte Carlo Sampling
#
# Creating an ensemble only requires the cluster expansion and a supercell matrix
# to define the sampling domain.

from smol.moca import CanonicalEnsemble
sc_matrix = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
ensemble = CanonicalEnsemble.from_cluster_expansion(

# # Running Monte Carlo sampling
#
# To generate MC samples for the ensemble, we need to create a sampler
# object.

from smol.moca import Sampler
sampler = Sampler.from_ensemble(

# In order to begin an MC simulation, an initial configuration must be provided.
# In this case we use pymatgen’s functionality to provide an ordered structure
# given a disordered one.

from pymatgen.transformations.standard_transformations import \
transformation = OrderDisorderedStructureTransformation()
structure = expansion.cluster_subspace.structure.copy()
structure.make_supercell(sc_matrix)
structure = transformation.apply_transformation(structure)

# Finally, the ordered structure can be used to generate an initial configuration
# to run MC sampling interations.

init_occu = ensemble.processor.occupancy_from_structure(structure)
sampler.run(1000000, initial_occupancy=init_occu)

# # Saving the generated objects and data
#
# To save the generated objects for the previous workflow we can simply use the
# provided convenience io functionaltiy. However, all main classes are
# serializable just as pymatgen and so can be saved as json dictionaries or
# using the [monty](https://guide.materialsvirtuallab.org/monty//) python
# package.

save_work(

# Example Notebooks
#
# For more detailed examples on how to use **smol** have a look at the following
# Jupyter notebooks.
#
# # Basic Examples
#
# * Creating a basic cluster expansion
#
# * Creating a cluster expansion with electrostatics
#
# * Visualizing clusters
#
# * Running Canonical Monte Carlo
#
# * Running Semi-Grand Canonical Monte Carlo
#
# * Preparing cluster expansion training data
#
# # Advanced Examples
#
# * Adding structures to a StructureWrangler in parallel
#
# * Simulated annealing with point electrostatics
#
# More to come…
