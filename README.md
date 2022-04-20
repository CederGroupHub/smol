<img src="docs/_static/logo.png" width="500px" alt="smol">

Statistical Mechanics on Lattices
=================================

![test](https://github.com/CederGroupHub/smol/actions/workflows/test.yml/badge.svg)

*Lighthweight but caffeinated Python implementation of computational methods
for statistical mechanical calculations of configurational states in
crystalline material systems.*

-----------------------------------------------------------------------------

> :warning: **smol** is still under substantial development and may possibly
> include changes that break backwards compatibility for the near future.

**smol** is a minimal implementation of computational methods to calculate
statistical mechanical and thermodynamic properties of crystalline
material systems based on the *cluster expansion* method from alloy theory and
related methods. Although **smol** is intentionally lightweight---in terms of
dependencies and built-in functionality---it has a modular design that closely
follows underlying mathematical formalism and provides useful abstractions to
easily extend existing methods or implement and test new ones. Finally,
although conceived mainly for method development, **smol** can (and is being)
used in production for materials science research applications.


Functionality
-------------
**smol** currently includes the following functionality:

- Defining cluster expansion functions for a given disordered structure using a
  variety of available site basis functions with and without explicit
  redundancy.
- Option to include explicit electrostatics in expansions using the Ewald
  summation method.
- Computing correlation vectors for a set of training structures with a variety
  of functionality to inspect the resulting feature matrix.
- Defining fitted cluster expansions for subsequent property prediction.
- Fast evaluation of correlation vectors and differences in correlation vectors
  from local updates in order to quickly compute properties and changes in
  properties for specified supercell sizes.
- Flexible toolset to sample cluster expansions using Monte Carlo with
  Canonical and Semigrand Canonical ensembles using a Metropolis sampler.

**smol** is built on top of [pymatgen](https://pymatgen.org) so any pre/post
structure analysis can be done seamlessly using the various functionality
supported there.

Installation
----------
`Clone` the repository. The latest tag in the `master` branch is the stable version of the
code. The `master` branch has the newest tested features, but may have more
lingering bugs.

Go to the top level directory of the cloned repo and type:

    pip install .

Usage
-----
Refer to the [documentation](http://amox.lbl.gov/smol) (requires lbl vpn) for details on using
**smol**. Going through the [example notebooks](https://github.com/CederGroupHub/smol/tree/master/examples)
will also help you get started.

Contributing
------------
We welcome all your contributions with open arms! Please fork and pull request any contributions.
See the
[contributing](https://github.com/CederGroupHub/smol/blob/master/CONTRIBUTING.md)
page in the documentation for how to contribute.


Changes
-------
The most recent changes are detailed in the
[change log](https://github.com/CederGroupHub/smol/blob/master/CHANGES.md).
