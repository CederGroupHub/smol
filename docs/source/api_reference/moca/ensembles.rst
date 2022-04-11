.. _ensembles:

=========
Ensembles
=========

This module contains classes for the different types of ensembles
that can be implemented for MC sampling. Systems at constant
temperature and constant composition (no species flow in/out of
the system) should use the :class:`CanonicalEnsemble`. Systems at
constant temperature and constant applied chemical potential
(species flow allowed) should use the :class:`SemiGrandEnsemble`.

.. toctree::
   :maxdepth: 2

   ensembles.base
   ensembles.canonical
   ensembles.semigrand
