---
title: 'smol: A Python package for cluster expansions and beyond'
tags:
  - Python
  - Cython
  - computational materials science
  - lattice models
  - cluster expansion method
  - Monte Carlo
authors:
  - name: Luis Barroso-Luque^[corresponding author]
    orcid: 0000-0002-6453-9545
    affiliation: "1, 2"
  - name: Julia H. Yang
    orcid: 0000-0002-5713-2288
    affiliation: "1, 2"
  - name: Fengyu Xie
    affiliation: "1, 2"
    orcid: 0000-0002-1169-1690
  - name: Tina Chen
    orcid: 0000-0003-0254-8339
    affiliation: "1, 2"
  - name: Ronald L. Kam
    affiliation: "1, 2"
  - name: Zinab Jadidi
    affiliation: "1, 2"
  - name: Peichen Zhong
    affiliation: "1, 2"
    orcid: 0000-0003-1921-1628
  - name: Gerbrand Ceder^[corresponding author]
    orcid: 0000-0001-9275-3605
    affiliation: "1, 2"
affiliations:
 - name: Department of Materials Science and Engineering, University of California Berkeley, Berkeley CA, 94720, USA
   index: 1
 - name: Materials Sciences Division, Lawrence Berkeley National Laboratory, Berkeley CA, 94720, USA
   index: 2
date: 12 April 2022
bibliography: paper.bib
---

# Summary

The growing research focus on multi-principal element materials—spanning a
variety of applications, such as electrochemical [@Lun:2020], structural
[@George:2019], semiconductor, thermoelectric, magnetic, and superconducting
[@Gao:2018] materials—necessitates the development of computational methodology
capable of resolving details of atomic configuration and resulting thermodynamic
properties. The cluster expansion method is a formal and effective way to
construct functions of atomic configuration by coarse-graining materials
properties, such as formation energies, in terms of species occupancy
lattice model [@Sanchez:1984]. The cluster expansion method coupled with Monte
Carlo sampling (CE-MC) is an established and effective way to resolve atomic
details underlying important thermodynamic properties [@VanderVen:2018].

`smol` (Statistical Mechanics on Lattices) is a Python package for constructing general applied lattice models, and performing Monte Carlo sampling of associated thermodynamic ensembles. The representation of lattice models in `smol` is based largely on the Cluster Expansion (CE) formalism [@Sanchez:1984]. However, the package is designed to allow easy implementation of extensions to the formalism, such as redundant representations [@Barroso-Luque:2021]. `smol` also includes flexible and extensible functionality to run Monte Carlo (MC) sampling from canonical and semigrand-canonical ensembles associated with the generated lattice models. `smol` has been intentionally designed to be lightweight and include a minimal set of dependencies to enable smooth installation, use, and development. `smol` was conceived primarily to enable development and implementation of novel CE-MC methodology but is now sufficiently mature that it is already being used in applied research of relevant material systems. [@Yang:2022-a; @Chen:2022; @Jadidi:2022; @Yang:2022-b]

# Statement of need

Several high-quality software packages implementing CE-MC methodology, such as `ATAT` [@VandeWalle:2002], `CASM` [@casm], `CLEASE` [@Chang:2019], and `icet` [@Angqvist:2019] are readily available either open source or by request. However, `smol` is distinct from existing CE-MC packages in both vision and implementation for the following three main reasons:

1. `smol` has been designed to easily develop, implement and test new methodology for the representation, fitting, and inference of applied lattice models beyond standard CE-MC methodology. The package has a heavily object-oriented and modular design that closely follows mathematical and methodological abstractions, which enables hassle-free implementation of methodology extensions. Furthermore, `smol` is written in pure Python (with a few critical components implemented in Cython to maintain performance) making it particularly developer friendly.

2. `smol` is the only package implemented using `pymatgen`—a widely used Python materials analysis library [@Ong:2013]. This allows seamless use of `pymatgen` functionality for pre and post-processing. Additionally, several other Materials Project [@Jain:2013] packages, such as Fireworks [@Jain:2015], atomate/atomate2 [@Mathew:2017; @atomate2], database creation, and management tools can be leveraged alongside `smol` to include configuration thermodynamic calculations as part of more elaborate materials analysis workflows.

3. `smol` is designed to be intentionally lightweight and dependency lean by delegating much of the non-core functionality to already well-established Python packages, for example, general structure manipulations, enumeration, and linear regression. This makes `smol` easy to install, easy to use, easy to develop, easy to extend and easy to test.

`smol` should be considerably more user and developer friendly than standalone C++ packages `ATAT` and `CASM`. In comparison to other Python implementations, in particular `icet`---which is superbly well documented and user-friendly---`smol` stands out as largely more developer friendly and easier to extend. In the context of all available packages, `smol` is geared towards efficient and open development of new methodology that is also user-friendly, thus allowing quick development-to-application turnaround time.

# Formalism overview

The atomic configuration of a crystalline material can be represented by a string of occupation variables, $\sigma = (\sigma_1, \sigma_2, \ldots, \sigma_N)$. Where the value of each occupation variable $\sigma_i$ represents the atomic species occupying the i-th site in an N-site supercell. Accordingly, any generalized lattice model of the atomic configuration can be written as a sum of multi-site (cluster) interaction functions,

\begin{equation}\label{eq:expansion}
H(\sigma) = \sum_{S\subseteq [N]} H_S(\sigma_S)
\end{equation}

Where $[N] = \{1, 2, , N\}$ is the set of all site indices, and $\sigma_S$ is the set of all occupation variables for the sites in a cluster $S$.

Two important considerations enable practical representations for effective fitting of applied lattice models:

1. A general procedure to construct function sets that span the function space over configurations.
2. Leveraging the symmetries of the underlying crystal structure to reduce the total function space to a subspace of symmetrically invariant functions only.

These two considerations are at the foundation of the original CE method [@Sanchez:1984], however, these considerations have been limited only to a small number of variations of the same formal representation. For example, consideration (1) has been limited only to a handful of different basis sets. In `smol` we have sought to implement a generalized version of the original CE method, where any symmetrically invariant lattice model is represented as,

\begin{equation}\label{eq:clusterexp}
H(\sigma) = \sum_\beta m_\beta J_\beta\Theta_\beta(\sigma)
\end{equation}

where $m_\beta$ are crystallographic multiplicities and $J_\beta$ are expansion coefficients. The correlation functions $\Theta_\beta$ take as input different sets of clusters of sites $S$ that are symmetrically equivalent under permutations corresponding to the symmetries of the underlying crystal structure’s space group. The set of all correlation functions $\{\Theta_\beta\}$, unlike the classical CE method, is not limited only to those that represent a basis set, but can be any complete set of functions (linearly independent or redundant) that spans the symmetry invariant function subspace over configurations $\sigma$.

Following the original CE method formalism, the correlation functions $\Theta_\beta$ are constructed from symmetrically adapted averages of cluster product functions,

\begin{align}
\Phi_\alpha(\sigma) &= \prod_{i=1}^N\phi_{\alpha_i}(\sigma_i) \label{eq:clusterfun}\\
\Theta_\beta(\sigma) &= \frac{1}{|\beta|}\sum_{\alpha\in\beta}\Phi_\alpha(\sigma)
\end{align}

Where the site functions $\phi_{\alpha_i}$ are single variable functions of each occupation variable; such that the set of all included site functions for each site $i$ span the associated space of all possible occupations of a given site. The multi-indices $\alpha$ ($|\alpha| = N$), serve as indices for the site function corresponding to each site in the supercell; and the orbits $\beta$ are sets of symmetrically equivalent multi-indices.

Functions represented by an appropriately truncated version of \autoref{eq:clusterexp} can be fitted to properties calculated with computationally intensive methods, such as first-principles electronic structure methods. The fitting procedure is predominantly done with linear regression using advanced regularization techniques. Subsequently, the resulting lattice model can be used in MC simulations to sample configurations for a corresponding statistical mechanical ensemble in order to efficiently compute thermodynamic functions and properties of atomic configuration.

# Package overview

The `smol` Python package is deliberately designed to be easily extensible and provide useful abstractions such that new methodology will rarely need to be implemented from scratch.

Classes and functions for representation and construction of functions of configuration (i.e. defining terms in a cluster expansion) are included in the `smol.cofe` module. Notably, the following object-oriented abstractions allow flexible definitions and the ability to easily implement extensions:

- Classes and functions to define site function sets, which make up the basic building blocks for an expansion as detailed in \autoref{eq:clusterfun}. The package includes functionality to generate both basis and redundant sets with any of the commonly used site function sets, (polynomial [@Sanchez:1984], trigonometric [@VandeWalle:2009], and occupancy indicator [@Zhang:2016]), as well as abstractions to effortlessly implement new function sets.
- Classes to represent clusters of sites S and groupings of symmetrically equivalent cluster functions to represent the terms in the sum of \autoref{eq:clusterexp}. Additionally, the package includes functionality to automatically generate these objects based on a given disordered structure—that may include neutral species, ionic species with assigned oxidation states, or vacancies—by leveraging `pymatgen`’s established and flexible representations of structures and associated symmetries.
- Classes to include additional interaction terms to a CE-based lattice model to improve training convergence. Currently, the package only includes an electrostatic pair potential for ionic structures [@Richards:2017], but the concept is applicable to any simple interaction model such as the reciprocal space CE constituent strain interaction [@Laks:1992], or any other empirical or fitted pair potential.
- Classes and functions to preprocess and generate feature matrices and fitting data corresponding to a defined set of correlation functions, datasets of relaxed structures, and computed energies from any first-principle, machine learning, or empirical potential calculations.

Additionally, the functionality to sample thermodynamic properties for a fitted lattice model under both canonical and semi-grand canonical ensembles is included in the `smol.moca` module. The `smol.moca` module includes flexible object-oriented abstractions, including the following:

- Classes and functions to quickly evaluate a cluster expansion for a given configuration and local configuration changes over a predefined supercell size and shape. Critical functions are implemented in Cython so that MC performance is not compromised.
- Classes to implement complex MC algorithms. The different components of MC are implemented as independent objects and utilities, that include classes to define configuration transition proposals, statistical ensembles, sampled value traces, and various Monte Carlo algorithm kernels. This enables customization of MC sampling methods, ensembles and computed properties without the need to re-write the sample generation, saving, and streaming to file functionality.

All classes and functions included in `smol` are thoroughly documented and several usage examples are available in the documentation.

# Acknowledgements

The development of `smol` was primarily funded by the U.S. Department of Energy, Office
of Science, Office of Basic Energy Sciences, Materials Sciences and Engineering Division
under Contract No. DE-AC02-05-CH11231 (Materials Project program KC23MP). L.B.L, Z.J., and T.C. also gratefully acknowledge support from the National Science Foundation Graduate Research Fellowship under Grant No. DGE 1752814 and DGE 1106400.

# References
