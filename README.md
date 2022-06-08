<img src="docs/_static/logo.png" width="500px" alt=" ">

Statistical Mechanics on Lattices
=================================


![test](https://github.com/CederGroupHub/smol/actions/workflows/test.yml/badge.svg)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/f6180b5223f346d2ac9dcf9a4bcc62d9)](https://www.codacy.com/gh/CederGroupHub/smol/dashboard?utm_source=github.com&utm_medium=referral&utm_content=CederGroupHub/smol&utm_campaign=Badge_Coverage)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/CederGroupHub/smol/main.svg)](https://results.pre-commit.ci/latest/github/CederGroupHub/smol/main)
[![pypi version](https://img.shields.io/pypi/v/smol?color=blue)](https://pypi.org/project/smol)
![python versions](https://img.shields.io/pypi/pyversions/smol)


*Lighthweight but caffeinated Python implementation of computational methods
for statistical mechanical calculations of configurational states in
crystalline material systems.*

-----------------------------------------------------------------------------

**smol** is a minimal implementation of computational methods to calculate
statistical mechanical and thermodynamic properties of crystalline
material systems based on the *cluster expansion* method from alloy theory and
related methods. Although **smol** is intentionally lightweight---in terms of
dependencies and built-in functionality---it has a modular design that closely
follows underlying mathematical formalism and provides useful abstractions to
easily extend existing methods or implement and test new ones. Finally,
although initially conceived for method development, **smol** can (and is being)
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
------------

> :warning: We have been granted the name `smol` on PyPi now. Please use `smol` instead
> of the previous alternative `statmech-on-lattices`.

From pypi:

    pip install smol

From source:

`Clone` the repository. The latest tag in the `main` branch is the stable version of the
code. The `main` branch has the newest tested features, but may have more
lingering bugs. From the top level directory

    pip install .

Usage
-----
Refer to the [documentation](https://cedergrouphub.github.io/smol/) for details on using
**smol**. Going through the [example notebooks](https://github.com/CederGroupHub/smol/tree/main/docs/src/notebooks)
will also help you get started.

Contributing
------------
We welcome all your contributions with open arms! Please fork and pull request any contributions.
See the
[developing](https://cedergrouphub.github.io/smol/developer_guide/index.html)
section in the documentation for how to contribute.


Changes
-------
The most recent changes are detailed in the
[change log](https://github.com/CederGroupHub/smol/blob/master/CHANGES.md).


Copyright Notice
----------------
    Statistical Mechanics on Lattices (smol) Copyright (c) 2022, The Regents
    of the University of California, through Lawrence Berkeley National
    Laboratory (subject to receipt of any required approvals from the U.S.
    Dept. of Energy) and the University of California, Berkeley. All rights reserved.

    If you have questions about your rights to use or distribute this software,
    please contact Berkeley Lab's Intellectual Property Office at
    IPO@lbl.gov.

    NOTICE.  This Software was developed under funding from the U.S. Department
    of Energy and the U.S. Government consequently retains certain rights.  As
    such, the U.S. Government has been granted for itself and others acting on
    its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
    Software to reproduce, distribute copies to the public, prepare derivative
    works, and perform publicly and display publicly, and to permit others to do so.
