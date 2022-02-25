==========
User Guide
==========

**smol** is a minimal implementation of computational methods to calculate
statistical mechanical and thermodynamic properties of crystalline
material systems based on the *cluster expansion* method from alloy theory and
related methods. Although **smol** is intentionally lightweight---in terms of
dependencies and built-in functionality---it has a modular design that closely
follows underlying mathematical formalism and provides useful abstractions to
easily extend existing methods or implement and test new ones. Finally,
although conceived mainly for method development, **smol** can (and is being)
used in production for materials science reasearch applications.

Functionality
=============
**smol** currently includes the following functionality:

- Defining cluster expansion terms for a given disordered structure using a
  variety of available site basis functions with and without explicit
  redundancy.
- Option to include explicit electrostatics using the Ewald summation method.
- Computing correlation vectors for a set of training structures with a variety
  of functionality to inspect the resulting feature matrix.
- Defining cluster expansions for subsequent property prediction.
- Fast evaluation of correlation vectors and differences in correlation vectors
  from local updates in order to quickly compute properties and changes in
  properties for specified supercells.
- Flexible toolset to sample cluster expansions using Monte Carlo with
  Canonical and Semigrand Canonical ensembles using a Metropolis sampler.

**smol** is built on top of `pymatgen <https://pymatgen.org/>`_ so any pre/post
structure analysis can be done seamlessly using the various functionality
supported there.